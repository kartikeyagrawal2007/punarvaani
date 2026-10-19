"""Audio-visual synchronisation check (SyncNet, Chung & Zisserman 2016).

Mined videos are often out of sync (dubbing, re-encoding, a different person
speaking off-screen, a news anchor's voice over B-roll). Training on those clips
teaches the model that lips and sound are unrelated. SyncNet embeds 0.2 s of
mouth video and 0.2 s of audio into a shared space where in-sync pairs are
close. For a clip we compare video at frame t with audio at t + δ for every
offset δ in ±15 frames:

* **offset** = argmin_δ mean distance (should be ~0; fix it if small, drop it if large);
* **confidence** = median(distances) − min(distances): a sharp minimum means
  the face we see *is* the voice we hear. LRS-style pipelines drop clips with
  confidence below ~3 (pretrained SyncNet scale).

This module holds the scoring algorithm and a SyncNet-like model; plug in the
pretrained ``syncnet_v2`` weights for production filtering.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class SyncResult:
    offset: int  # frames; positive = audio lags video
    confidence: float
    distances: torch.Tensor  # (2 * max_offset + 1,)

    def ok(self, min_confidence: float = 3.0, max_abs_offset: int = 3) -> bool:
        return self.confidence >= min_confidence and abs(self.offset) <= max_abs_offset


def av_offset(video_emb: torch.Tensor, audio_emb: torch.Tensor, max_offset: int = 15) -> SyncResult:
    """``video_emb``, ``audio_emb``: (T, D) per-frame embeddings at 25 Hz."""
    t = min(video_emb.shape[0], audio_emb.shape[0])
    v, a = video_emb[:t], audio_emb[:t]
    padded = F.pad(a.T, (max_offset, max_offset)).T  # zero audio outside the clip
    dists = []
    for off in range(-max_offset, max_offset + 1):
        shifted = padded[max_offset + off : max_offset + off + t]
        valid = slice(max(0, -off), t - max(0, off))  # only frames where shifted audio is real
        dists.append(torch.linalg.vector_norm(v[valid] - shifted[valid], dim=-1).mean())
    d = torch.stack(dists)
    best = int(d.argmin())
    return SyncResult(best - max_offset, float(d.median() - d.min()), d)


class SyncNet(nn.Module):
    """Two-stream network: 5 mouth frames (0.2 s) vs. 20 MFCC/mel frames (0.2 s) -> embeddings."""

    def __init__(self, dim: int = 256, n_mels: int = 40, width: int = 64):
        super().__init__()
        w = width
        self.video = nn.Sequential(
            nn.Conv3d(1, w, (5, 7, 7), (1, 2, 2), (0, 3, 3)), nn.BatchNorm3d(w), nn.ReLU(),
            nn.Conv3d(w, 2 * w, (1, 5, 5), (1, 2, 2), (0, 2, 2)), nn.BatchNorm3d(2 * w), nn.ReLU(),
            nn.Conv3d(2 * w, 4 * w, (1, 3, 3), (1, 2, 2), (0, 1, 1)), nn.BatchNorm3d(4 * w), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(4 * w, dim),
        )  # fmt: skip
        self.audio = nn.Sequential(
            nn.Conv2d(1, w, 3, 1, 1), nn.BatchNorm2d(w), nn.ReLU(),
            nn.Conv2d(w, 2 * w, 3, (2, 1), 1), nn.BatchNorm2d(2 * w), nn.ReLU(),
            nn.Conv2d(2 * w, 4 * w, 3, 2, 1), nn.BatchNorm2d(4 * w), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4 * w, dim),
        )  # fmt: skip

    def embed_video(self, windows: torch.Tensor) -> torch.Tensor:  # (N, 5, H, W)
        return F.normalize(self.video(windows.unsqueeze(1)), dim=-1)

    def embed_audio(self, windows: torch.Tensor) -> torch.Tensor:  # (N, n_mels, 20)
        return F.normalize(self.audio(windows.unsqueeze(1)), dim=-1)

    @torch.no_grad()
    def clip_embeddings(self, video: torch.Tensor, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-frame embeddings for a clip: video (T, H, W), mel (n_mels, 4T)."""
        t = min(video.shape[0], mel.shape[1] // 4)
        vpad = F.pad(video[:t].unsqueeze(0), (0, 0, 0, 0, 2, 2), mode="replicate")[0]
        mpad = F.pad(mel[:, : 4 * t], (8, 8), mode="replicate")
        vw = torch.stack([vpad[i : i + 5] for i in range(t)])
        aw = torch.stack([mpad[:, 4 * i : 4 * i + 20] for i in range(t)])
        return self.embed_video(vw), self.embed_audio(aw)


def contrastive_sync_loss(v: torch.Tensor, a: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE over a batch of (video window, audio window) pairs: matching pairs on the diagonal.

    (The original SyncNet used a margin loss on pairs; InfoNCE trains faster.)
    """
    logits = v @ a.T / temperature
    labels = torch.arange(v.shape[0], device=v.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

"""ECAPA-TDNN speaker embeddings (Desplanques et al., Interspeech 2020).

A speaker embedding ("voice fingerprint") is a fixed-length vector that is
close for two clips of the same person and far for different people. We use
it to (1) find which segments of a family video are the patient (vs. relatives),
(2) condition the Path B vocoder on the patient's voice, and (3) measure voice
similarity in evaluation.

Architecture: TDNN = 1-D convolutions over time on log-mel frames.

* SE-Res2Blocks: Res2Net splits channels into 8 groups processed hierarchically
  (multi-scale receptive fields in one layer); squeeze-excitation rescales
  channels using global context; dilations 2, 3, 4 widen the time context.
* Multi-layer feature aggregation: concatenate all blocks' outputs (low layers
  carry useful speaker detail too).
* Attentive statistics pooling: a per-frame attention weight (computed with
  global mean/std as context) gives a weighted mean *and* standard deviation -
  so the embedding focuses on informative frames and ignores silence/noise.
* Trained with AAM-softmax (additive angular margin): classify training
  speakers but require a margin ``m`` in angle, which makes embeddings
  separate cleanly under cosine similarity.

For real use, train this on VoxCeleb + Hindi speakers or port SpeechBrain's
pretrained ``spkrec-ecapa-voxceleb`` weights.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from punarvaani.preprocessing.audio import LogMel


class SEBlock(nn.Module):
    def __init__(self, ch: int, bottleneck: int = 128):
        super().__init__()
        self.fc1 = nn.Conv1d(ch, bottleneck, 1)
        self.fc2 = nn.Conv1d(bottleneck, ch, 1)

    def forward(self, x):
        s = x.mean(-1, keepdim=True)
        return x * torch.sigmoid(self.fc2(F.relu(self.fc1(s))))


class Res2Conv(nn.Module):
    def __init__(self, ch: int, kernel: int, dilation: int, scale: int = 8):
        super().__init__()
        if ch % scale:
            raise ValueError("channels must be divisible by scale")
        self.scale, w = scale, ch // scale
        self.convs = nn.ModuleList(nn.Conv1d(w, w, kernel, dilation=dilation, padding=dilation * (kernel - 1) // 2)
                                   for _ in range(scale - 1))  # fmt: skip
        self.bns = nn.ModuleList(nn.BatchNorm1d(w) for _ in range(scale - 1))

    def forward(self, x):
        chunks = x.chunk(self.scale, 1)
        out, y = [chunks[0]], None
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            y = chunks[i + 1] if y is None else chunks[i + 1] + y
            y = bn(F.relu(conv(y)))
            out.append(y)
        return torch.cat(out, 1)


class SERes2Block(nn.Module):
    def __init__(self, ch: int, kernel: int, dilation: int, scale: int = 8):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(ch, ch, 1), nn.ReLU(), nn.BatchNorm1d(ch))
        self.res2 = Res2Conv(ch, kernel, dilation, scale)
        self.conv2 = nn.Sequential(nn.Conv1d(ch, ch, 1), nn.ReLU(), nn.BatchNorm1d(ch))
        self.se = SEBlock(ch, max(8, ch // 4))

    def forward(self, x):
        return x + self.se(self.conv2(self.res2(self.conv1(x))))


class AttentiveStatsPool(nn.Module):
    def __init__(self, ch: int, attn: int = 128):
        super().__init__()
        self.attention = nn.Sequential(nn.Conv1d(3 * ch, attn, 1), nn.ReLU(), nn.BatchNorm1d(attn), nn.Tanh(),
                                       nn.Conv1d(attn, ch, 1))  # fmt: skip

    def forward(self, x, mask=None):  # x: (B, C, T); mask: (B, T) True = valid
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[-1], dtype=torch.bool, device=x.device)
        m = mask.unsqueeze(1).float()
        n = m.sum(-1, keepdim=True).clamp(min=1)
        mean = (x * m).sum(-1, keepdim=True) / n
        std = (((x - mean) ** 2 * m).sum(-1, keepdim=True) / n).clamp(min=1e-6).sqrt()
        ctx = torch.cat([x, mean.expand_as(x), std.expand_as(x)], 1)
        w = self.attention(ctx).masked_fill(~mask.unsqueeze(1), -1e4).softmax(-1)
        mu = (w * x).sum(-1)
        sg = ((w * x**2).sum(-1) - mu**2).clamp(min=1e-6).sqrt()
        return torch.cat([mu, sg], 1)


class ECAPATDNN(nn.Module):
    def __init__(
        self, n_mels: int = 80, channels: int = 512, embed_dim: int = 192, mfa_channels: int | None = None
    ):
        super().__init__()
        mfa = mfa_channels or 3 * channels
        self.fbank = LogMel(n_fft=512, win_length=400, hop_length=160, n_mels=n_mels)
        self.layer1 = nn.Sequential(
            nn.Conv1d(n_mels, channels, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(channels)
        )
        self.blocks = nn.ModuleList(SERes2Block(channels, 3, d) for d in (2, 3, 4))
        self.mfa = nn.Sequential(nn.Conv1d(3 * channels, mfa, 1), nn.ReLU())
        self.pool = AttentiveStatsPool(mfa)
        self.bn = nn.BatchNorm1d(2 * mfa)
        self.fc = nn.Linear(2 * mfa, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, S) 16 kHz waveform -> (B, embed_dim) embedding (not normalised)."""
        x = self.fbank(wav)
        x = x - x.mean(-1, keepdim=True)  # cepstral mean normalisation: removes channel/mic colouring
        x = self.layer1(x)
        outs = []
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        x = self.mfa(torch.cat(outs, 1))
        return self.fc(self.bn(self.pool(x)))

    @classmethod
    def tiny(cls) -> ECAPATDNN:
        return cls(n_mels=40, channels=32, embed_dim=32, mfa_channels=48)

    @torch.no_grad()
    def embed(self, wav: torch.Tensor) -> torch.Tensor:
        """L2-normalised embedding for one waveform (S,)."""
        self.eval()
        return F.normalize(self(wav.unsqueeze(0))[0], dim=-1)


class AAMSoftmax(nn.Module):
    """Additive angular margin softmax: logit for the true class is s·cos(θ + m)."""

    def __init__(self, embed_dim: int, num_speakers: int, margin: float = 0.2, scale: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_speakers, embed_dim) * 0.01)
        self.m, self.s = margin, scale

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cos = F.linear(F.normalize(emb), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        target = torch.cos(theta + self.m)
        # keep the function monotonic when θ + m > π
        target = torch.where(theta + self.m < math.pi, target, cos - self.m * math.sin(self.m))
        one_hot = F.one_hot(labels, cos.shape[1]).bool()
        logits = self.s * torch.where(one_hot, target, cos)
        return F.cross_entropy(logits, labels)

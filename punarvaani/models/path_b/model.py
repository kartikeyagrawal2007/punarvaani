"""Path B: lips -> discrete speech units (+ mel, + pitch) every 20 ms, no text in between.

Speech units are cluster IDs of a self-supervised speech model's features
(mHuBERT-147 layer features + k-means, K≈1000). They are a "phonetic alphabet
learned from audio": they capture *what* is said and *how it's timed*, but
mostly not *who* says it - speaker identity is added back by the vocoder. So
predicting units from lips turns a hard regression (video -> waveform) into a
classification problem per 20 ms frame, which is far easier to train and gives
intelligible speech (Choi, Kim & Ro, Interspeech 2023, "lip2speech-unit").

Heads on top of the shared visual encoder (25 Hz):

* **unit head** - upsample 2x to 50 Hz, small conv stack, softmax over K units;
* **mel head** - 4 mel frames per video frame (100 Hz). lip2speech-unit found
  that predicting mel *and* units ("multi-target") helps both;
* **pitch head** (optional) - log-F0 + voiced/unvoiced per unit frame, so the
  vocoder gets intonation from the lips instead of a flat robotic pitch;
* **auxiliary CTC** over characters (optional) - content supervision from the
  transcript keeps the encoder focused on linguistic content.

The time alignment is free: units are extracted from the *same clip's* audio,
so frame t of video corresponds to units 2t and 2t+1. No duration model needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from punarvaani.models.encoder.avhubert import VisualSpeechEncoder
from punarvaani.utils import lengths_to_padding_mask


class ConvBlock(nn.Module):
    def __init__(self, dim: int, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel, padding=kernel // 2)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # (B, T, D)
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x + self.drop(F.gelu(self.norm(y)))


class UpsampleHead(nn.Module):
    """(B, T, D) -> (B, T * factor, out_dim)."""

    def __init__(self, dim: int, hidden: int, out_dim: int, factor: int, blocks: int = 2):
        super().__init__()
        self.factor, self.hidden = factor, hidden
        self.up = nn.Linear(dim, hidden * factor)
        self.blocks = nn.Sequential(*[ConvBlock(hidden) for _ in range(blocks)])
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x):
        b, t, _ = x.shape
        y = self.up(x).view(b, t * self.factor, self.hidden)
        return self.out(self.blocks(y))


@dataclass
class LipToUnitOutput:
    loss: torch.Tensor | None
    logs: dict
    unit_logits: torch.Tensor  # (B, 2T, K)
    mel: torch.Tensor | None  # (B, 4T, n_mels)
    pitch: torch.Tensor | None  # (B, 2T, 2): log-f0 (normalised), voiced logit
    ctc_log_probs: torch.Tensor | None  # (B, T, V)
    padding_mask: torch.Tensor  # (B, T)


class LipToUnitModel(nn.Module):
    def __init__(self, encoder: VisualSpeechEncoder, num_units: int = 1000, hidden: int = 512, n_mels: int = 80,
                 predict_mel: bool = True, predict_pitch: bool = False, ctc_vocab_size: int | None = None,
                 mel_weight: float = 1.0, pitch_weight: float = 0.5, ctc_weight: float = 0.1,
                 label_smoothing: float = 0.1):  # fmt: skip
        super().__init__()
        d = encoder.dim
        self.encoder = encoder
        self.num_units = num_units
        self.unit_head = UpsampleHead(d, hidden, num_units, factor=2)
        self.mel_head = UpsampleHead(d, hidden, n_mels, factor=4, blocks=1) if predict_mel else None
        self.pitch_head = UpsampleHead(d, hidden // 2, 2, factor=2, blocks=1) if predict_pitch else None
        self.ctc_head = nn.Linear(d, ctc_vocab_size) if ctc_vocab_size else None
        self.mel_weight, self.pitch_weight, self.ctc_weight = mel_weight, pitch_weight, ctc_weight
        self.label_smoothing = label_smoothing

    def forward(self, video, lengths, units=None, mel=None, pitch=None, tokens=None, token_lengths=None,
                return_features: bool = False) -> LipToUnitOutput:  # fmt: skip
        mask = lengths_to_padding_mask(lengths, video.shape[1])
        feats = self.encoder(video, mask)
        unit_logits = self.unit_head(feats)
        mel_pred = self.mel_head(feats) if self.mel_head is not None else None
        pitch_pred = self.pitch_head(feats) if self.pitch_head is not None else None
        ctc_lp = self.ctc_head(feats).float().log_softmax(-1) if self.ctc_head is not None else None

        loss, logs = None, {}
        if units is not None:
            loss = F.cross_entropy(unit_logits.float().transpose(1, 2), units, ignore_index=-100,
                                   label_smoothing=self.label_smoothing)  # fmt: skip
            valid = units != -100
            logs["unit_ce"] = loss.detach()
            logs["unit_acc"] = ((unit_logits.argmax(-1) == units) & valid).sum() / valid.sum().clamp(min=1)
            if mel_pred is not None and mel is not None:
                mel_mask = ~lengths_to_padding_mask(lengths * 4, mel_pred.shape[1])
                l1 = (mel_pred.float() - mel).abs().mean(-1)
                mel_loss = (l1 * mel_mask).sum() / mel_mask.sum().clamp(min=1)
                loss = loss + self.mel_weight * mel_loss
                logs["mel_l1"] = mel_loss.detach()
            if pitch_pred is not None and pitch is not None:
                # pitch: (B, 2T, 2) = [normalised log-f0, voiced in {0,1}]; f0 loss only on voiced frames
                voiced = pitch[..., 1]
                frame_mask = valid.float()
                vuv = F.binary_cross_entropy_with_logits(pitch_pred[..., 1].float(), voiced, reduction="none")
                f0 = (pitch_pred[..., 0].float() - pitch[..., 0]).pow(2)
                p_loss = (vuv * frame_mask).sum() / frame_mask.sum().clamp(min=1) + (
                    f0 * voiced * frame_mask
                ).sum() / (voiced * frame_mask).sum().clamp(min=1)
                loss = loss + self.pitch_weight * p_loss
                logs["pitch"] = p_loss.detach()
            if ctc_lp is not None and tokens is not None:
                ctc = F.ctc_loss(ctc_lp.transpose(0, 1), tokens, lengths, token_lengths, zero_infinity=True,
                                 reduction="sum") / tokens.shape[0]  # fmt: skip
                loss = loss + self.ctc_weight * ctc
                logs["ctc"] = ctc.detach()
            logs["loss"] = loss.detach()
        out = LipToUnitOutput(loss, logs, unit_logits, mel_pred, pitch_pred, ctc_lp, mask)
        if return_features:
            out.features = feats  # type: ignore[attr-defined]
        return out

    @torch.no_grad()
    def predict_units(
        self, video: torch.Tensor, length: int | None = None, temperature: float = 0.0
    ) -> torch.Tensor:
        """Units for one clip (T, H, W) -> (2T,). ``temperature`` 0 = argmax."""
        length = length or video.shape[0]
        out = self(video.unsqueeze(0), torch.tensor([length], device=video.device))
        logits = out.unit_logits[0, : 2 * length]
        if temperature <= 0:
            return logits.argmax(-1)
        return torch.distributions.Categorical(logits=logits / temperature).sample()

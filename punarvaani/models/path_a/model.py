"""Path A recogniser: visual encoder -> Hindi characters, trained with joint CTC + attention.

Why both losses (Watanabe et al. 2017, "hybrid CTC/attention")?

* **CTC** assumes output tokens are conditionally independent given the video
  and forces a monotonic alignment (speech moves forward in time). That makes
  it robust and gives frame-level posteriors, which phrase mode and the
  hybrid path need. But it cannot model spelling/grammar dependencies.
* The **attention decoder** is a conditional language model over characters
  (it sees previous characters), so it fixes spelling, but on its own it can
  loop or stop early on hard inputs.

Training on ``λ·CTC + (1-λ)·attention`` regularises the encoder with CTC's
monotonic alignment and gives us both scorers at test time: CTC prefix beam
search proposes an n-best list, the decoder re-ranks it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from punarvaani.models.encoder.avhubert import VisualSpeechEncoder
from punarvaani.models.encoder.transformer import MultiheadAttention
from punarvaani.utils import lengths_to_padding_mask


class SinusoidalPositions(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[1]]


class DecoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.self_attn = MultiheadAttention(dim, heads, dropout)
        self.cross_attn = MultiheadAttention(dim, heads, dropout)
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, dim)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, y, memory, memory_mask, tgt_mask):
        y = y + self.drop(self.self_attn(self.ln1(y), key_padding_mask=tgt_mask, causal=True))
        y = y + self.drop(self.cross_attn(self.ln2(y), memory, key_padding_mask=memory_mask))
        return y + self.drop(self.ffn(self.ln3(y)))


class TransformerDecoder(nn.Module):
    def __init__(
        self, vocab_size: int, dim: int, layers: int, heads: int, ffn_dim: int, dropout: float = 0.1
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = SinusoidalPositions(dim)
        self.layers = nn.ModuleList(DecoderLayer(dim, heads, ffn_dim, dropout) for _ in range(layers))
        self.ln = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, vocab_size, bias=False)
        self.out.weight = self.embed.weight  # tied input/output embeddings
        self.scale = math.sqrt(dim)

    def forward(self, tokens, memory, memory_mask=None, tgt_mask=None):
        y = self.pos(self.embed(tokens) * self.scale)
        for layer in self.layers:
            y = layer(y, memory, memory_mask, tgt_mask)
        return self.out(self.ln(y))


@dataclass
class VSROutput:
    loss: torch.Tensor | None
    ctc_loss: torch.Tensor | None
    att_loss: torch.Tensor | None
    ctc_log_probs: torch.Tensor  # (B, T, V)
    encoder_out: torch.Tensor  # (B, T, D)
    padding_mask: torch.Tensor  # (B, T) True = pad


class VSRModel(nn.Module):
    def __init__(
        self,
        encoder: VisualSpeechEncoder,
        vocab_size: int,
        blank_id: int = 0,
        pad_id: int = 1,
        bos_id: int = 2,
        eos_id: int = 2,
        decoder_layers: int = 6,
        decoder_heads: int | None = None,
        decoder_ffn: int | None = None,
        decoder_dropout: float = 0.1,
        ctc_weight: float = 0.3,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        d = encoder.dim
        self.encoder = encoder
        self.ctc_head = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, vocab_size))
        self.decoder = (
            TransformerDecoder(vocab_size, d, decoder_layers, decoder_heads or encoder.cfg.heads,
                               decoder_ffn or 4 * d, decoder_dropout)
            if decoder_layers > 0 else None
        )  # fmt: skip
        self.blank_id, self.pad_id, self.bos_id, self.eos_id = blank_id, pad_id, bos_id, eos_id
        self.ctc_weight = ctc_weight if self.decoder is not None else 1.0
        self.label_smoothing = label_smoothing

    def encode(self, video: torch.Tensor, lengths: torch.Tensor):
        mask = lengths_to_padding_mask(lengths, video.shape[1])
        enc = self.encoder(video, mask)
        return enc, mask

    def forward(self, video, lengths, targets=None, target_lengths=None) -> VSROutput:
        """``targets``: (B, L) padded with ``pad_id``; no BOS/EOS (added here)."""
        enc, mask = self.encode(video, lengths)
        log_probs = self.ctc_head(enc).float().log_softmax(-1)
        if targets is None:
            return VSROutput(None, None, None, log_probs, enc, mask)

        ctc = F.ctc_loss(
            log_probs.transpose(0, 1), targets, lengths, target_lengths,
            blank=self.blank_id, reduction="sum", zero_infinity=True,
        ) / targets.shape[0]  # fmt: skip
        att = torch.zeros((), device=enc.device)
        if self.decoder is not None and self.ctc_weight < 1.0:
            dec_in, dec_out = self.make_decoder_io(targets, target_lengths)
            logits = self.decoder(dec_in, enc, mask, dec_in == self.pad_id)
            att = F.cross_entropy(
                logits.float().transpose(1, 2), dec_out, ignore_index=-100,
                label_smoothing=self.label_smoothing, reduction="sum",
            ) / targets.shape[0]  # fmt: skip
        loss = self.ctc_weight * ctc + (1 - self.ctc_weight) * att
        return VSROutput(loss, ctc, att, log_probs, enc, mask)

    def make_decoder_io(self, targets: torch.Tensor, target_lengths: torch.Tensor):
        b = targets.shape[0]
        bos = torch.full((b, 1), self.bos_id, dtype=targets.dtype, device=targets.device)
        dec_in = torch.cat([bos, targets], 1)
        dec_out = torch.cat([targets, torch.full_like(bos, self.pad_id)], 1).clone()
        dec_out[torch.arange(b), target_lengths] = self.eos_id
        pad = lengths_to_padding_mask(target_lengths + 1, dec_out.shape[1])
        dec_out = dec_out.masked_fill(pad, -100)
        dec_in = dec_in.masked_fill(lengths_to_padding_mask(target_lengths + 1, dec_in.shape[1]), self.pad_id)
        return dec_in, dec_out

    @torch.no_grad()
    def decoder_score(
        self, enc: torch.Tensor, enc_mask: torch.Tensor | None, hyps: list[list[int]]
    ) -> torch.Tensor:
        """Log P_att(hyp + <eos> | video) for each hypothesis (teacher forced). ``enc`` is (1, T, D)."""
        if self.decoder is None or not hyps:
            return torch.zeros(len(hyps))
        lens = torch.tensor([len(h) for h in hyps], device=enc.device)
        targets = torch.full(
            (len(hyps), max(1, int(lens.max()))), self.pad_id, dtype=torch.long, device=enc.device
        )
        for i, h in enumerate(hyps):
            targets[i, : len(h)] = torch.tensor(h, dtype=torch.long)
        dec_in, dec_out = self.make_decoder_io(targets, lens)
        memory = enc.expand(len(hyps), -1, -1)
        mem_mask = enc_mask.expand(len(hyps), -1) if enc_mask is not None else None
        logp = self.decoder(dec_in, memory, mem_mask, dec_in == self.pad_id).float().log_softmax(-1)
        tok = logp.gather(-1, dec_out.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return (tok * (dec_out != -100)).sum(-1).cpu()

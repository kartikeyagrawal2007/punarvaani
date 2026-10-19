"""Hybrid A+B: Path A's text guides Path B's unit predictions (stretch goal).

Path B (lips -> units) keeps natural timing and prosody but makes word errors;
Path A (lips -> text -> LLM) gets words right but sounds flat. LipVoicer
(ICLR 2024) combines them for a diffusion model: at each denoising step it
nudges the generated mel toward what an ASR model says matches the predicted
text ("classifier guidance"). We do the same in the *discrete unit* space:

1. a small **unit ASR** model maps a (soft) unit sequence to character CTC
   posteriors. It is trained on audio only: units from mHuBERT, text from the
   transcript - plenty of data, no video needed;
2. at inference, take Path B's unit logits ``z`` (50 Hz) and Path A's final
   text ``y``. Repeat a few times::

       p   = softmax(z / τ)                     # differentiable "soft units"
       L   = CTC(unit_asr(p), y)                # how badly do these units spell y?
       z  ← z − η · ∇_z L                       # move units toward spelling y

   plus a trust-region term ``λ·||z − z₀||²`` so we only fix *content* and keep
   Path B's timing (the original frame structure).
3. argmax(z) goes to the unit vocoder as usual.

Because the CTC alignment is monotonic and re-computed every step, the text
can only pull units locally: "बानी" frames become "पानी" frames, the pause
before them stays where the lips put it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class UnitASR(nn.Module):
    """Soft units (B, T, K) -> character log-probs (B, T', V). T' = T // 2 (back to 25 Hz)."""

    def __init__(self, num_units: int, vocab_size: int, dim: int = 256, layers: int = 3):
        super().__init__()
        self.unit_embed = nn.Linear(num_units, dim, bias=False)  # = embedding lookup for one-hot input
        self.convs = nn.ModuleList(nn.Conv1d(dim, dim, 5, padding=2) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))
        self.downsample = nn.Conv1d(dim, dim, 2, stride=2)
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, soft_units: torch.Tensor) -> torch.Tensor:
        x = self.unit_embed(soft_units)
        for conv, norm in zip(self.convs, self.norms):
            x = x + F.gelu(norm(conv(x.transpose(1, 2)).transpose(1, 2)))
        x = self.downsample(x.transpose(1, 2)).transpose(1, 2)
        return self.out(x).log_softmax(-1)

    def loss(self, units: torch.Tensor, unit_lengths: torch.Tensor, tokens: torch.Tensor, token_lengths: torch.Tensor,
             num_units: int, blank: int = 0) -> torch.Tensor:  # fmt: skip
        """Training loss from hard units (B, T) with -100 padding."""
        one_hot = F.one_hot(units.clamp(min=0), num_units).float()
        lp = self(one_hot)
        return F.ctc_loss(lp.transpose(0, 1), tokens, unit_lengths // 2, token_lengths, blank=blank,
                          zero_infinity=True)  # fmt: skip


@dataclass
class GuidanceResult:
    units: torch.Tensor  # (T,) guided hard units
    logits: torch.Tensor  # (T, K) guided logits
    ctc_before: float
    ctc_after: float
    changed_fraction: float


def guide_units(unit_logits: torch.Tensor, tokens: torch.Tensor, unit_asr: UnitASR, steps: int = 50,
                step_size: float = 1.0, temperature: float = 1.0, trust: float = 0.05, blank: int = 0) -> GuidanceResult:  # fmt: skip
    """Gradient guidance of one utterance's unit logits (T, K) toward the character sequence ``tokens``.

    Plain (unnormalised) gradient steps: a globally normalised step oscillated
    against the trust term and spread edits over the whole utterance, while
    plain steps make small, local edits where the text disagrees (on the toy
    test: CTC 7.0 -> 0.1 nats with 10% of frames changed, timing untouched).
    """
    z0 = unit_logits.detach().float()
    z = z0.clone().requires_grad_(True)
    target = tokens.view(1, -1)
    t_out = torch.tensor([z.shape[0] // 2])
    t_len = torch.tensor([tokens.numel()])

    def ctc(zz):
        lp = unit_asr(torch.softmax(zz / temperature, -1).unsqueeze(0))
        return F.ctc_loss(lp.transpose(0, 1), target, t_out, t_len, blank=blank, reduction="sum")

    was_training = unit_asr.training
    unit_asr.eval()
    for p in unit_asr.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        before = float(ctc(z0))
    for _ in range(steps):
        loss = ctc(z) + trust * (z - z0).pow(2).sum()
        (grad,) = torch.autograd.grad(loss, z)
        with torch.no_grad():
            z -= step_size * grad
    with torch.no_grad():
        after = float(ctc(z))
    for p in unit_asr.parameters():
        p.requires_grad_(True)
    unit_asr.train(was_training)
    units = z.detach().argmax(-1)
    changed = float((units != z0.argmax(-1)).float().mean())
    return GuidanceResult(units, z.detach(), before, after, changed)

"""Pitch (F0) tracking with YIN (de Cheveigné & Kawahara 2002), vectorised in torch.

Path B can predict intonation from the lips (emphasis and question rise move
the jaw and lips too); its targets come from the training audio's F0. YIN:

1. difference function d(τ) = Σ (x_t - x_{t+τ})² for each lag τ;
2. cumulative-mean normalisation d'(τ) = d(τ) / ((1/τ) Σ_{j≤τ} d(j)), which
   removes the bias toward τ = 0 and makes a fixed threshold meaningful;
3. the first lag where d' dips below the threshold (else the global minimum)
   is the period; F0 = sample_rate / τ. A high minimum means "unvoiced".

Frames are 20 ms (50 Hz) to line up with speech units.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from punarvaani.preprocessing.audio import SAMPLE_RATE


def yin(wav: torch.Tensor, sample_rate: int = SAMPLE_RATE, hop: int = 320, frame: int = 1024,
        fmin: float = 60.0, fmax: float = 450.0, threshold: float = 0.15) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    """Returns (f0_hz, voiced) per hop; f0 is 0 where unvoiced."""
    tau_min, tau_max = int(sample_rate / fmax), int(sample_rate / fmin)
    n_frames = wav.numel() // hop
    wav = F.pad(wav, (frame // 2, frame // 2 + tau_max))
    frames = wav.unfold(0, frame + tau_max, hop)[:n_frames]  # (N, frame + tau_max)
    x = frames[:, :frame]
    # d(τ) via energies and cross-correlation (FFT), exact for each τ
    energy = F.pad(frames.pow(2).cumsum(1), (1, 0))
    e0 = energy[:, frame : frame + 1]
    e_tau = energy[:, frame : frame + tau_max + 1] - energy[:, : tau_max + 1]
    n_fft = 1 << (2 * frame + tau_max).bit_length()
    xc = torch.fft.irfft(torch.fft.rfft(frames, n_fft) * torch.fft.rfft(x, n_fft).conj(), n_fft)[
        :, : tau_max + 1
    ]
    d = (e0 + e_tau - 2 * xc).clamp(min=0)
    d[:, 0] = 0
    cmndf = torch.ones_like(d)
    cum = d[:, 1:].cumsum(1)
    taus = torch.arange(1, tau_max + 1, device=wav.device)
    cmndf[:, 1:] = d[:, 1:] * taus / cum.clamp(min=1e-12)

    search = cmndf[:, tau_min : tau_max + 1]
    below = search < threshold
    first = torch.where(below.any(1), below.float().argmax(1), search.argmin(1))
    # walk to the local minimum after the first threshold crossing
    idx = first.clone()
    for _ in range(tau_max - tau_min):
        nxt = (idx + 1).clamp(max=search.shape[1] - 1)
        better = search.gather(1, nxt[:, None]).squeeze(1) < search.gather(1, idx[:, None]).squeeze(1)
        if not better.any():
            break
        idx = torch.where(better, nxt, idx)
    best = search.gather(1, idx[:, None]).squeeze(1)
    tau = (idx + tau_min).float()
    # parabolic interpolation for sub-sample precision
    left = search.gather(1, (idx - 1).clamp(min=0)[:, None]).squeeze(1)
    right = search.gather(1, (idx + 1).clamp(max=search.shape[1] - 1)[:, None]).squeeze(1)
    denom = left - 2 * best + right
    tau = tau + torch.where(denom.abs() > 1e-9, 0.5 * (left - right) / denom, torch.zeros_like(tau))
    voiced = (best < threshold * 2) & (x.pow(2).mean(1) > 1e-6)
    f0 = torch.where(voiced, sample_rate / tau, torch.zeros_like(tau))
    return f0, voiced


def pitch_targets(f0: torch.Tensor, voiced: torch.Tensor, mean_log_f0: float | None = None,
                  std_log_f0: float | None = None) -> torch.Tensor:  # fmt: skip
    """(N, 2): speaker-normalised log-F0 (0 when unvoiced) and the voiced flag.

    Normalising per speaker means the lip model predicts *relative* intonation
    (rise/fall) and the vocoder maps it into the patient's own pitch range.
    """
    log_f0 = torch.log(f0.clamp(min=1.0))
    v = voiced.bool()
    mean = mean_log_f0 if mean_log_f0 is not None else (log_f0[v].mean().item() if v.any() else 0.0)
    std = std_log_f0 if std_log_f0 is not None else (log_f0[v].std().item() if v.sum() > 1 else 1.0)
    norm = torch.where(v, (log_f0 - mean) / max(std, 1e-3), torch.zeros_like(log_f0))
    return torch.stack([norm, v.float()], -1)

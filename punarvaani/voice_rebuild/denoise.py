"""Denoising and quality scoring of archival voice notes.

WhatsApp voice notes and family videos are recorded on phones in kitchens,
streets and hospital rooms, then compressed (Opus ~16-24 kbit/s). Before we can
clone a voice from them we need (a) less noise, (b) to know which bits are
good enough to use.

* :func:`deepfilternet_enhance` - DeepFilterNet3 (Schröter et al. 2023), a
  small real-time speech-enhancement network. Best quality; optional dependency.
* :func:`spectral_wiener` - dependency-free fallback: estimate the noise
  spectrum from the quietest frames, apply a smoothed Wiener gain with a floor.
  The floor matters: removing noise completely creates "musical noise"
  artefacts that a voice cloner would faithfully copy.
* :func:`quality_report` - SNR estimate, clipping, effective bandwidth and
  speech duration, combined into one score used to rank segments.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from punarvaani.preprocessing.audio import SAMPLE_RATE, energy_vad


def deepfilternet_enhance(wav: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Enhance with DeepFilterNet (``pip install deepfilternet``); it runs at 48 kHz internally."""
    from df.enhance import enhance, init_df

    from punarvaani.preprocessing.audio import resample

    model, state, _ = init_df()
    x = resample(wav, sample_rate, state.sr()).unsqueeze(0)
    y = enhance(model, state, x)[0]
    return resample(y, state.sr(), sample_rate)


def spectral_wiener(wav: torch.Tensor, n_fft: int = 512, hop: int = 128, noise_percentile: float = 0.1,
                    gain_floor: float = 0.1, over_subtraction: float = 1.5, smooth_frames: int = 3) -> torch.Tensor:  # fmt: skip
    window = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(wav, n_fft, hop, window=window, return_complex=True)
    power = spec.abs() ** 2  # (F, T)
    frame_energy = power.sum(0)
    k = max(1, int(noise_percentile * frame_energy.numel()))
    quiet = frame_energy.topk(k, largest=False).indices
    noise = power[:, quiet].mean(1, keepdim=True)
    gain = (1 - over_subtraction * noise / power.clamp(min=1e-10)).clamp(min=gain_floor)
    if smooth_frames > 1:  # smooth over time to avoid musical noise
        gain = F.avg_pool1d(gain.unsqueeze(0), smooth_frames, 1, smooth_frames // 2, count_include_pad=False)[
            0
        ]
        gain = gain[:, : power.shape[1]]
    out = torch.istft(spec * gain, n_fft, hop, window=window, length=wav.numel())
    return out


def denoise(wav: torch.Tensor, sample_rate: int = SAMPLE_RATE, backend: str = "auto") -> torch.Tensor:
    if backend in ("auto", "deepfilternet"):
        try:
            return deepfilternet_enhance(wav, sample_rate)
        except ImportError:
            if backend == "deepfilternet":
                raise
    return spectral_wiener(wav)


@dataclass
class QualityReport:
    snr_db: float
    clipping_ratio: float
    bandwidth_hz: float
    speech_seconds: float
    score: float


def estimate_bandwidth(
    wav: torch.Tensor, sample_rate: int = SAMPLE_RATE, threshold_db: float = -50.0
) -> float:
    """Highest frequency whose average power is within ``threshold_db`` of the peak.

    Telephone-band audio (8 kHz sampled, upsampled) shows ~4 kHz here and makes
    a poor TTS reference - it would teach the cloner a muffled voice.
    """
    spec = (
        torch.stft(wav, 1024, 256, window=torch.hann_window(1024), return_complex=True).abs().pow(2).mean(1)
    )
    db = 10 * torch.log10(spec.clamp(min=1e-12))
    above = torch.nonzero(db > db.max() + threshold_db).squeeze(1)
    return float(above.max()) * sample_rate / 1024 if above.numel() else 0.0


def estimate_snr_db(wav: torch.Tensor, frame: int = 400, hop: int = 160) -> float:
    """VAD-free SNR: noise power = 10th-percentile frame power, speech = 90th percentile minus noise.

    Archival notes nearly always contain pauses, so the quietest frames are a
    good noise estimate even when a VAD would be fooled by steady noise.
    """
    if wav.numel() < frame:
        wav = F.pad(wav, (0, frame - wav.numel()))
    p = wav.unfold(0, frame, hop).pow(2).mean(1)
    noise = torch.quantile(p, 0.1).clamp(min=1e-10)
    speech = (torch.quantile(p, 0.9) - noise).clamp(min=1e-10)
    return float(10 * torch.log10(speech / noise))


def quality_report(wav: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> QualityReport:
    snr = min(estimate_snr_db(wav), 60.0)
    clip = float((wav.abs() > 0.99).float().mean())
    bw = estimate_bandwidth(wav, sample_rate)
    speech_s = float(energy_vad(wav).sum()) * 0.01
    # Score: SNR matters most; heavy penalty for clipping; narrowband and very short audio penalised.
    score = (
        min(snr, 40.0) / 40.0 - 20.0 * clip - (0.3 if bw < 5000 else 0.0) - (0.3 if speech_s < 1.0 else 0.0)
    )
    return QualityReport(snr, clip, bw, speech_s, score)

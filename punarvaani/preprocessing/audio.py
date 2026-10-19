"""Audio front-end shared by the data engine, Path B targets, the vocoder and ECAPA.

Frame-rate bookkeeping (the single most common source of bugs in AV models):

    video          25 fps   -> 40 ms per frame
    speech units   50 Hz    -> 20 ms, 2 per video frame (mHuBERT / HuBERT stride 320 @16 kHz)
    mel / fbank   100 Hz    -> 10 ms, 4 per video frame (hop 160 @16 kHz)
    waveform     16 kHz     -> 640 samples per video frame

Everything here uses 16 kHz mono. The mel filterbank is implemented directly on
``torch.stft`` so results do not depend on the installed torchaudio version.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SAMPLE_RATE = 16000
VIDEO_FPS = 25
HOP_LENGTH = 160
SAMPLES_PER_VIDEO_FRAME = SAMPLE_RATE // VIDEO_FPS  # 640
MELS_PER_VIDEO_FRAME = SAMPLES_PER_VIDEO_FRAME // HOP_LENGTH  # 4
UNITS_PER_VIDEO_FRAME = 2


def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Load any file soundfile can read as mono float32 at ``sample_rate``."""
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(axis=1))
    return resample(wav, sr, sample_rate)


def save_audio(path: str | Path, wav: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> None:
    import soundfile as sf

    sf.write(str(path), wav.detach().cpu().float().clamp(-1, 1).numpy(), sample_rate)


def resample(wav: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    if orig_sr == new_sr:
        return wav
    try:
        import torchaudio.functional as AF

        return AF.resample(wav, orig_sr, new_sr)
    except (ImportError, AttributeError, RuntimeError):
        # Band-limit with a windowed-sinc low-pass, then interpolate.
        n = round(wav.shape[-1] * new_sr / orig_sr)
        if new_sr < orig_sr:
            wav = _lowpass(wav, cutoff=0.5 * new_sr / orig_sr)
        return F.interpolate(wav.view(1, 1, -1), size=n, mode="linear", align_corners=False).view(-1)


def _lowpass(wav: torch.Tensor, cutoff: float, taps: int = 101) -> torch.Tensor:
    t = torch.arange(taps) - (taps - 1) / 2
    h = 2 * cutoff * torch.sinc(2 * cutoff * t) * torch.hann_window(taps, periodic=False)
    h = h / h.sum()
    return F.conv1d(wav.view(1, 1, -1), h.view(1, 1, -1), padding=taps // 2).view(-1)


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int, n_fft: int, sample_rate: int, fmin: float = 0.0, fmax: float | None = None
) -> torch.Tensor:
    """Triangular HTK-style mel filters, shape (n_mels, n_fft // 2 + 1), area-normalised."""
    fmax = fmax or sample_rate / 2
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    fb = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        lo, mid, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (bins - lo) / max(mid - lo, 1e-8)
        down = (hi - bins) / max(hi - mid, 1e-8)
        fb[i] = np.maximum(0, np.minimum(up, down))
    fb *= (2.0 / (hz[2:] - hz[:-2]))[:, None]  # Slaney area normalisation
    return torch.from_numpy(fb).float()


class LogMel(torch.nn.Module):
    """Log-mel spectrogram at 100 Hz (hop 160). Defaults match lip2speech-unit / HiFi-GAN @16 kHz."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = 640,
        win_length: int = 640,
        hop_length: int = HOP_LENGTH,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: float | None = 8000.0,
        center: bool = True,
    ):
        super().__init__()
        self.n_fft, self.win_length, self.hop_length, self.center = n_fft, win_length, hop_length, center
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.register_buffer("fb", mel_filterbank(n_mels, n_fft, sample_rate, fmin, fmax), persistent=False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, S) or (S,) waveform -> (B, n_mels, frames) natural-log mel."""
        squeeze = wav.dim() == 1
        wav = wav.unsqueeze(0) if squeeze else wav
        if not self.center:
            # HiFi-GAN convention: reflect-pad so frames = S // hop exactly.
            pad = (self.n_fft - self.hop_length) // 2
            wav = F.pad(wav.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spec = torch.stft(
            wav,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window,
            center=self.center,
            return_complex=True,
        ).abs()
        mel = torch.log(torch.clamp(self.fb @ spec, min=1e-5))
        return mel.squeeze(0) if squeeze else mel


def stack_frames(feats: torch.Tensor, n: int = MELS_PER_VIDEO_FRAME) -> torch.Tensor:
    """(T, D) at 100 Hz -> (T // n, D * n) at 25 Hz, as AV-HuBERT does with its 26-d fbank (-> 104-d)."""
    t = feats.shape[0] // n * n
    return feats[:t].reshape(t // n, -1)


def rms_normalize(wav: torch.Tensor, target_dbfs: float = -23.0, peak: float = 0.99) -> torch.Tensor:
    rms = wav.pow(2).mean().sqrt().clamp(min=1e-8)
    wav = wav * (10 ** (target_dbfs / 20) / rms)
    return wav * min(1.0, peak / max(wav.abs().max().item(), 1e-8))


def frame_energy_db(wav: torch.Tensor, frame: int = 400, hop: int = HOP_LENGTH) -> torch.Tensor:
    if wav.numel() < frame:
        wav = F.pad(wav, (0, frame - wav.numel()))
    frames = wav.unfold(0, frame, hop)
    return 10 * torch.log10(frames.pow(2).mean(-1).clamp(min=1e-10))


def energy_vad(wav: torch.Tensor, threshold_db: float = -40.0, relative_db: float = 30.0, min_speech_ms: int = 200,
               noise_margin_db: float = 6.0) -> torch.Tensor:  # fmt: skip
    """Boolean speech mask at 100 Hz.

    A frame is speech if it is louder than (a) an absolute floor, (b) the noise
    floor (10th percentile frame energy) plus ``noise_margin_db``, and (c)
    ``relative_db`` below the loudest frame. (b) is what makes it work on
    voice notes with steady background noise (fans, traffic). Short blips are
    removed.
    """
    e = frame_energy_db(wav)
    floor = torch.quantile(e, 0.1)
    speech = (e > threshold_db) & (e > floor + noise_margin_db) & (e > e.max() - relative_db)
    min_frames = max(1, min_speech_ms // 10)
    out = speech.clone()
    start = None
    for i, s in enumerate([*speech.tolist(), False]):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if i - start < min_frames:
                out[start:i] = False
            start = None
    return out


def num_frames(num_samples: int, rate_hz: int) -> int:
    return math.floor(num_samples * rate_hz / SAMPLE_RATE)

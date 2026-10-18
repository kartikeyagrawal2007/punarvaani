"""Text -> speech in the user's own voice (Path A's "newsreader").

IndicF5 (AI4Bharat) is an F5-TTS model trained on Indian languages. F5-TTS is a
flow-matching (diffusion-like) model that does *zero-shot voice cloning*: given a
reference clip and its transcript, it continues "in that voice". No per-user
training is needed - the reference comes from :mod:`punarvaani.voice_rebuild`
(a clean 5-12 s segment of an old voice note plus its IndicWhisper transcript).

Phrase mode never waits for synthesis: :class:`CachedTTS` renders all of the
user's phrases once, offline, so a recognised phrase plays instantly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from punarvaani.text.hindi import normalize


class TTSBackend(Protocol):
    sample_rate: int

    def synthesize(self, text: str) -> torch.Tensor: ...


class IndicF5TTS:
    """Wrapper around ``ai4bharat/IndicF5`` (Hugging Face, ``trust_remote_code``).

    Usage follows the model card: ``model(text, ref_audio_path=..., ref_text=...)``
    returns a numpy waveform at 24 kHz. Check the card if the interface changes.
    """

    sample_rate = 24000

    def __init__(
        self,
        ref_audio_path: str | Path,
        ref_text: str,
        repo_id: str = "ai4bharat/IndicF5",
        device: str | None = None,
    ):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
        if device:
            self.model = self.model.to(device)
        self.ref_audio_path, self.ref_text = str(ref_audio_path), ref_text

    def synthesize(self, text: str) -> torch.Tensor:
        audio = self.model(normalize(text), ref_audio_path=self.ref_audio_path, ref_text=self.ref_text)
        audio = np.asarray(audio)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        return torch.from_numpy(audio.astype(np.float32))


class ToneTTS:
    """Deterministic stand-in for tests and demos without a GPU: one beep per character."""

    sample_rate = 16000

    def synthesize(self, text: str) -> torch.Tensor:
        text = normalize(text)
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        freqs = 200 + (np.frombuffer(text.encode(), dtype=np.uint8)[: max(1, len(text))] + seed % 7) * 2.0
        t = np.arange(int(0.06 * self.sample_rate)) / self.sample_rate
        wav = np.concatenate([0.2 * np.sin(2 * np.pi * f * t) for f in freqs]) if len(text) else np.zeros(1)
        return torch.from_numpy(wav.astype(np.float32))


class CachedTTS:
    """Memoises synthesis in memory and (optionally) on disk, keyed by normalised text."""

    def __init__(self, backend: TTSBackend, cache_dir: str | Path | None = None):
        self.backend = backend
        self.sample_rate = backend.sample_rate
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, torch.Tensor] = {}

    def _path(self, key: str) -> Path | None:
        return self.cache_dir / f"{hashlib.sha1(key.encode()).hexdigest()}.pt" if self.cache_dir else None

    def synthesize(self, text: str) -> torch.Tensor:
        key = normalize(text)
        if key in self._mem:
            return self._mem[key]
        path = self._path(key)
        if path is not None and path.exists():
            wav = torch.load(path, weights_only=True)
        else:
            wav = self.backend.synthesize(key)
            if path is not None:
                torch.save(wav, path)
        self._mem[key] = wav
        return wav

    def prerender(self, phrases) -> int:
        for p in phrases:
            self.synthesize(p)
        return len(self._mem)

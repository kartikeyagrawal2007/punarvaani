"""A synthetic "toy lip-reading" world for smoke tests, debugging and the CPU demo.

Real data is large, private and slow; bugs in shapes, masks, CTC lengths or
LoRA wiring should be caught in seconds. Here each letter of a tiny alphabet is
a *mouth shape* (an ellipse with a letter-specific height, width and tilt);
an utterance is a sequence of letters, each held 2-4 frames with smooth
transitions, closed mouth between words, plus noise and jitter. A model that
can learn this has working plumbing; one that can't has a bug.

``confusable`` letter pairs share one shape - a built-in viseme ambiguity, so
the effect of a language model / phrase prior can be demonstrated.

Style knobs (``open_scale``, ``width_scale``, ``tilt_offset``) shift how the
same letters look, which simulates the silent-mouthing / new-speaker domain gap
that per-user LoRA must close.

It also emits Path B targets: a unit id per 20 ms (letter id, 0 for silence)
and a per-letter tone as "audio".
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from punarvaani.models.encoder.avhubert import normalize_video
from punarvaani.text.vocab import CharVocab

DEFAULT_ALPHABET = "कमलपनसतरगद"
DEFAULT_WORDS = ("कमल", "पतन", "नल", "सरस", "मन", "दल", "गगन", "तरल", "पद", "नगर", "सपन", "मगर")


def _shape_params(
    alphabet: str, confusable: Sequence[tuple[str, str]] = ()
) -> dict[str, tuple[float, float, float]]:
    params = {}
    n = len(alphabet)
    for i, ch in enumerate(alphabet):
        a = 2 * math.pi * i / n
        params[ch] = (
            0.15 + 0.3 * (0.5 + 0.5 * math.sin(a)),
            0.25 + 0.2 * (0.5 + 0.5 * math.cos(a)),
            0.25 * math.sin(2 * a),
        )
    for a, b in confusable:
        params[b] = params[a]
    return params


def render_mouth(
    height: float, width: float, tilt: float, size: int, rng: np.random.Generator, noise: float
) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size - 0.5
    xs = xs + rng.normal(0, 0.01)
    ys = ys + rng.normal(0, 0.01)
    c, s = math.cos(tilt), math.sin(tilt)
    xr, yr = c * xs + s * ys, -s * xs + c * ys
    outer = (xr / max(width, 1e-3)) ** 2 + (yr / max(height + 0.06, 1e-3)) ** 2
    inner = (xr / max(width * 0.8, 1e-3)) ** 2 + (yr / max(height, 1e-3)) ** 2
    img = 0.55 * np.ones((size, size), dtype=np.float32)
    img[outer < 1] = 0.8  # lips
    img[inner < 1] = 0.1  # dark mouth opening
    img += rng.normal(0, noise, img.shape)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


class SyntheticLipDataset(Dataset):
    def __init__(self, num_samples: int = 512, size: int = 44, alphabet: str = DEFAULT_ALPHABET,
                 words: Sequence[str] = DEFAULT_WORDS, max_words: int = 3, noise: float = 0.05,
                 confusable: Sequence[tuple[str, str]] = (), seed: int = 0, sentences: Sequence[str] | None = None,
                 open_scale: float = 1.0, width_scale: float = 1.0, tilt_offset: float = 0.0):  # fmt: skip
        self.vocab = CharVocab.from_symbols(alphabet)
        self.words = [w for w in words if all(ch in alphabet for ch in w)]
        self.params = _shape_params(alphabet, confusable)
        self.n, self.size, self.max_words, self.noise, self.seed = num_samples, size, max_words, noise, seed
        self.sentences = list(sentences) if sentences else None
        # Articulation style. Silent mouthing is typically *hypo-articulated*: smaller
        # openings, less lip spreading. open_scale < 1 simulates that; tilt_offset
        # simulates a different speaker's habitual mouth posture.
        self.style = (open_scale, width_scale, tilt_offset)

    def __len__(self) -> int:
        return self.n

    def sentence(self, i: int) -> str:
        if self.sentences:
            return self.sentences[i % len(self.sentences)]
        rng = random.Random(self.seed * 100003 + i)
        return " ".join(rng.choice(self.words) for _ in range(rng.randint(1, self.max_words)))

    def render(self, text: str, rng: np.random.Generator) -> tuple[np.ndarray, list[int]]:
        """Return video (T, size, size) uint8 and per-frame letter ids (0 = silence)."""
        keyframes: list[tuple[tuple[float, float, float], int, int]] = [((0.0, 0.3, 0.0), 2, 0)]
        for k, word in enumerate(text.split()):
            if k:
                keyframes.append(((0.0, 0.3, 0.0), 2, 0))
            for ch in word:
                keyframes.append((self.params[ch], int(rng.integers(2, 5)), self.vocab.index[ch]))
        keyframes.append(((0.0, 0.3, 0.0), 2, 0))
        frames, labels = [], []
        prev = keyframes[0][0]
        for shape, dur, label in keyframes:
            for f in range(dur):
                a = min(1.0, (f + 1) / 2)  # 2-frame transition from the previous shape
                h, w, t = (prev[j] * (1 - a) + shape[j] * a for j in range(3))
                h, w, t = h * self.style[0], w * self.style[1], t + self.style[2]
                frames.append(render_mouth(h, w, t, self.size, rng, self.noise))
                labels.append(label)
            prev = shape
        return np.stack(frames), labels

    def __getitem__(self, i: int) -> dict:
        text = self.sentence(i)
        rng = np.random.default_rng(self.seed * 7919 + i)
        video, labels = self.render(text, rng)
        units = torch.tensor(labels, dtype=torch.long).repeat_interleave(2)  # 50 Hz
        return {
            "id": f"synthetic_{i}",
            "video": normalize_video(torch.from_numpy(video)),
            "text": text,
            "tokens": torch.tensor(self.vocab.encode(text, already_normalized=True), dtype=torch.long),
            "units": units,
            "mel": letter_tones_mel(labels),
            "mode": "spoken",
        }


def letter_tones_mel(labels: Sequence[int], n_mels: int = 80) -> torch.Tensor:
    """A fake log-mel (4 frames per video frame): one bright band per letter."""
    mel = torch.full((4 * len(labels), n_mels), -8.0)
    for t, lab in enumerate(labels):
        if lab:
            band = (lab * 7) % (n_mels - 4)
            mel[4 * t : 4 * t + 4, band : band + 4] = 0.0
    return mel


def letter_tones_wav(labels: Sequence[int], sample_rate: int = 16000, fps: int = 25) -> torch.Tensor:
    """Fake speech for the toy world: each letter is a harmonic tone, silence is quiet noise.

    Used to train/smoke-test the unit vocoder: units (letter ids at 50 Hz) -> this audio.
    """
    n = sample_rate // fps
    t = np.arange(n) / sample_rate
    rng = np.random.default_rng(0)
    chunks = []
    for lab in labels:
        if lab:
            f0 = 110.0 + 15.0 * lab
            chunks.append(0.3 * np.sin(2 * np.pi * f0 * t) + 0.1 * np.sin(2 * np.pi * 2 * f0 * t))
        else:
            chunks.append(0.003 * rng.standard_normal(n))
    return torch.from_numpy(np.concatenate(chunks).astype(np.float32))

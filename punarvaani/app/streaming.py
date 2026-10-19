"""Real-time pieces: causal lip cropping, lip-activity endpointing, streaming Path A and Path B.

Offline preprocessing may look into the future (centred smoothing, whole-clip
interpolation). Live, we can't, so:

* :class:`OnlineLipCropper` smooths landmarks with a *causal* exponential
  moving average (a little lag instead of look-ahead);
* :class:`LipActivityDetector` decides when an utterance starts and ends from
  mouth motion - silent speakers produce no audio, so a normal voice-activity
  detector is useless. Hysteresis (separate on/off thresholds plus a hang-over
  time) stops it from chopping words at short closures like "प" or "म";
* :class:`UtteranceStreamer` buffers crops while the person is speaking and
  runs Path A on each finished utterance (latency = end-pointing hang-over +
  recognition time);
* :class:`ChunkedUnitStreamer` runs Path B every ``chunk`` frames on a window
  with ``left`` context and ``right`` look-ahead, emitting audio continuously
  (latency ≈ (chunk + right) frames + compute, e.g. (8 + 4) × 40 ms ≈ 0.5 s).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from punarvaani.models.encoder.avhubert import normalize_video
from punarvaani.preprocessing.lip_crop import (
    CANONICAL_STABLE,
    LIPS_INNER,
    LIPS_OUTER,
    STABLE_POINTS,
    _warp_affine,
    apply_transform,
    similarity_transform,
    to_gray,
)


class OnlineLipCropper:
    def __init__(self, crop_size: int = 96, alpha: float = 0.5, out_size: int = 88):
        self.crop_size, self.alpha, self.out_size = crop_size, alpha, out_size
        self._lm: np.ndarray | None = None
        self._centre: np.ndarray | None = None

    def reset(self) -> None:
        self._lm = self._centre = None

    def __call__(self, frame_rgb: np.ndarray, landmarks: np.ndarray | None) -> np.ndarray | None:
        """One frame in, one (out_size, out_size) uint8 centre crop out (None until a face is seen)."""
        if landmarks is not None:
            self._lm = landmarks if self._lm is None else self.alpha * landmarks + (1 - self.alpha) * self._lm
        if self._lm is None:
            return None  # hold the last landmarks through short detection drop-outs
        m = similarity_transform(self._lm[list(STABLE_POINTS)], CANONICAL_STABLE)
        centre = apply_transform(self._lm[list(LIPS_OUTER + LIPS_INNER)], m).mean(0)
        self._centre = (
            centre if self._centre is None else self.alpha * centre + (1 - self.alpha) * self._centre
        )
        m = m.copy()
        m[:, 2] += self.crop_size / 2 - self._centre
        crop = to_gray(_warp_affine(frame_rgb, m, self.crop_size))
        o = (self.crop_size - self.out_size) // 2
        return crop[o : o + self.out_size, o : o + self.out_size]


@dataclass
class ActivityEvent:
    kind: str  # "start" or "end"
    frame: int


class LipActivityDetector:
    """Hysteresis end-pointer on mouth-motion energy (mean absolute frame difference of crops)."""

    def __init__(self, on_ratio: float = 3.0, off_ratio: float = 1.8, min_on_frames: int = 3, hang_frames: int = 12,
                 floor_window: int = 75):  # fmt: skip
        self.on_ratio, self.off_ratio = on_ratio, off_ratio
        self.min_on, self.hang = min_on_frames, hang_frames
        self.history: deque[float] = deque(maxlen=floor_window)
        self.prev: np.ndarray | None = None
        self.active, self.on_count, self.off_count, self.t = False, 0, 0, -1

    def floor(self) -> float:
        # Noise floor = a low percentile of recent motion: camera noise, breathing, small head motion.
        return max(float(np.percentile(self.history, 20)) if self.history else 0.0, 0.5)

    def __call__(self, crop: np.ndarray) -> ActivityEvent | None:
        self.t += 1
        motion = 0.0 if self.prev is None else float(np.abs(crop.astype(np.float32) - self.prev).mean())
        self.prev = crop.astype(np.float32)
        floor = self.floor()
        if not self.active:
            self.history.append(motion)
        self.on_count = self.on_count + 1 if motion > self.on_ratio * floor else 0
        self.off_count = self.off_count + 1 if motion < self.off_ratio * floor else 0
        if not self.active and self.on_count >= self.min_on:
            self.active, self.off_count = True, 0
            return ActivityEvent("start", self.t - self.min_on + 1)
        if self.active and self.off_count >= self.hang:
            self.active, self.on_count = False, 0
            return ActivityEvent("end", self.t - self.hang + 1)
        return None


class UtteranceStreamer:
    """Feeds crops through the detector; calls ``on_utterance(video)`` with each complete utterance."""

    def __init__(self, on_utterance: Callable[[torch.Tensor], object], detector: LipActivityDetector | None = None,
                 pre_roll: int = 5, max_frames: int = 375):  # fmt: skip
        self.on_utterance = on_utterance
        self.detector = detector or LipActivityDetector()
        self.pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll)  # lips move just before the detector fires
        self.buffer: list[np.ndarray] = []
        self.max_frames = max_frames
        self.results: list = []

    def push(self, crop: np.ndarray):
        event = self.detector(crop)
        if self.detector.active or (event and event.kind == "end"):
            if not self.buffer:
                self.buffer.extend(self.pre_roll)
            self.buffer.append(crop)
        else:
            self.pre_roll.append(crop)
        if (event and event.kind == "end") or len(self.buffer) >= self.max_frames:
            video = normalize_video(torch.from_numpy(np.stack(self.buffer)))
            self.buffer = []
            result = self.on_utterance(video)
            self.results.append(result)
            return result
        return None


class ChunkedUnitStreamer:
    """Streaming Path B: encoder over [left context | chunk | look-ahead], keep the chunk's units."""

    def __init__(self, unit_model, vocoder=None, speaker: torch.Tensor | None = None, chunk: int = 8, left: int = 50,
                 right: int = 4):  # fmt: skip
        self.model, self.vocoder, self.speaker = unit_model.eval(), vocoder, speaker
        self.chunk, self.left, self.right = chunk, left, right
        self.frames: list[torch.Tensor] = []
        self.emitted = 0  # video frames whose units were already emitted

    @property
    def latency_frames(self) -> int:
        return self.chunk + self.right

    @torch.no_grad()
    def _run(self, end: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        start = max(0, self.emitted - self.left)
        stop = min(len(self.frames), end + self.right)
        window = torch.stack(self.frames[start:stop])
        logits = self.model(window.unsqueeze(0), torch.tensor([window.shape[0]])).unit_logits[0]
        units = logits[2 * (self.emitted - start) : 2 * (end - start)].argmax(-1)
        self.emitted = end
        audio = None
        if self.vocoder is not None:
            spk = self.speaker.unsqueeze(0) if self.speaker is not None else None
            audio = self.vocoder(units.unsqueeze(0), spk)[0, 0]
        return units, audio

    def push(self, frame: torch.Tensor):
        """``frame``: one normalised (H, W) crop. Returns (units, audio) when a chunk is ready."""
        self.frames.append(frame)
        if len(self.frames) - self.emitted >= self.chunk + self.right:
            return self._run(self.emitted + self.chunk)
        return None

    def flush(self):
        if self.emitted < len(self.frames):
            return self._run(len(self.frames))
        return None

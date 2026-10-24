"""A cartoon face that mouths phrases, so the lab can be tried (and tested) without a webcam.

It produces both the video frame and MediaPipe-style landmarks (478 points, of
which the 5 rigid points and 40 lip points are meaningful), so everything after
the landmarker - crop, lip-activity detection, features, classifiers - runs
exactly as it does on a real face. Each phrase gets its own fixed sequence of
mouth shapes; every take varies speed, size, head position and noise.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable

import numpy as np

from punarvaani.preprocessing.lip_crop import CANONICAL_STABLE, LIPS_INNER, LIPS_OUTER, STABLE_POINTS

_MOUTH_CENTRE = np.array([128.0, 188.0])  # in the canonical 256x256 face


def phrase_shapes(text: str) -> np.ndarray:
    """(K, 2) per-syllable (opening, width) targets, fixed for a given phrase."""
    rng = np.random.default_rng(zlib.crc32(text.encode("utf-8")))
    k = 2 + len(text) % 4 + rng.integers(0, 2)
    return np.stack([rng.uniform(2, 20, k), rng.uniform(38, 62, k)], 1)


def mouth_track(text: str, rng: np.random.Generator, rest=(1.0, 50.0)) -> np.ndarray:
    """(T, 2) smooth (opening, width) trajectory for one take of ``text``."""
    shapes = phrase_shapes(text) * rng.uniform(0.9, 1.1, (1, 2))
    per = max(3, int(round(6 * rng.uniform(0.85, 1.15))))
    keys = np.vstack([rest, shapes, rest])
    t_key = np.arange(len(keys)) * per
    t = np.arange(t_key[-1] + 1)
    track = np.stack([np.interp(t, t_key, keys[:, d]) for d in range(2)], 1)
    return track + rng.normal(0, 0.3, track.shape)


def lip_points(opening: float, width: float) -> np.ndarray:
    """(40, 2) canonical lip points in MediaPipe order (20 outer, then 20 inner)."""

    def ring(w, h_top, h_bottom):
        bottom = np.linspace(np.pi, 0, 11)  # left corner -> bottom middle -> right corner
        top = np.linspace(0, np.pi, 11)[1:-1]  # right -> top middle -> left (corners already used)
        pts = [(w / 2 * np.cos(a), h_bottom * np.sin(a)) for a in bottom]
        pts += [(w / 2 * np.cos(a), -h_top * np.sin(a)) for a in top]
        return np.array(pts)

    outer = ring(width, 7 + 0.3 * opening, 9 + 0.7 * opening)
    inner = ring(0.8 * width, opening / 2, opening / 2)
    return np.vstack([outer, inner]) + _MOUTH_CENTRE


class SyntheticFace:
    """Video source: ``read()`` gives the next RGB frame, ``landmarks(frame)`` its 478 points.

    ``next_phrase()`` is asked what to say each time the face starts a new
    utterance (the lab passes the phrase it is currently prompting for).
    """

    fps = 25.0

    def __init__(self, next_phrase: Callable[[], str | None], size=(640, 480), seed: int = 0):
        self.next_phrase, (self.w, self.h) = next_phrase, size
        self.rng = np.random.default_rng(seed)
        self.t, self.track, self.pause = 0, None, 30
        self.spoken: list[str] = []  # what the face actually said, for tests
        self._lm: np.ndarray | None = None

    def _head(self) -> np.ndarray:
        """2x3 canonical->image transform with a slow head bob and sway."""
        s = 1.25 + 0.05 * np.sin(self.t / 40)
        a = 0.05 * np.sin(self.t / 55)
        rot = s * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        shift = np.array([self.w / 2, self.h / 2 + 10]) - rot @ np.array([128.0, 140.0])
        shift += [12 * np.sin(self.t / 33), 6 * np.sin(self.t / 47)]
        return np.hstack([rot, shift[:, None]])

    def _mouth(self) -> tuple[float, float]:
        if self.track is None:
            self.pause -= 1
            if self.pause <= 0:
                phrase = self.next_phrase()
                if phrase:
                    self.track, self.k = mouth_track(phrase, self.rng), 0
                    self.spoken.append(phrase)
                else:
                    self.pause = 10
            return 1.0 + self.rng.normal(0, 0.1), 50.0
        o, w = self.track[self.k]
        self.k += 1
        if self.k >= len(self.track):
            self.track, self.pause = None, int(self.rng.integers(25, 40))
        return float(o), float(w)

    def read(self) -> np.ndarray:
        import cv2

        self.t += 1
        m = self._head()
        lm = np.zeros((478, 2))
        stable = CANONICAL_STABLE @ m[:, :2].T + m[:, 2]
        lm[list(STABLE_POINTS)] = stable
        lips = lip_points(*self._mouth()) @ m[:, :2].T + m[:, 2]
        lm[list(LIPS_OUTER + LIPS_INNER)] = lips
        self._lm = lm + self.rng.normal(0, 0.15, lm.shape)

        frame = np.full((self.h, self.w, 3), (28, 46, 52), np.uint8)
        face_c = (np.array([128.0, 150.0]) @ m[:, :2].T + m[:, 2]).astype(int)
        cv2.ellipse(
            frame, tuple(face_c), (int(95 * m[0, 0]), int(125 * m[0, 0])), 0, 0, 360, (224, 186, 160), -1
        )
        for x, y in stable[:4].reshape(2, 2, 2).mean(1):
            cv2.circle(frame, (int(x), int(y)), int(9 * m[0, 0]), (40, 40, 40), -1)
        nose = stable[4].astype(int)
        cv2.circle(frame, tuple(nose), int(5 * m[0, 0]), (200, 150, 130), -1)
        cv2.fillPoly(frame, [lips[:20].astype(np.int32)], (170, 70, 70))
        cv2.fillPoly(frame, [lips[20:].astype(np.int32)], (50, 15, 20))
        noise = self.rng.normal(0, 2.0, frame.shape)
        return np.clip(frame + noise, 0, 255).astype(np.uint8)

    def landmarks(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        return self._lm

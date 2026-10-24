"""Lip-shape features from face landmarks, for the few-shot phrase readers in the lab.

The real lip reader works on pixels (88x88 crops) and needs hours of data. With
only a handful of takes per phrase, a far smaller input works better: the 40
lip landmarks themselves. Per frame we

1. fit the same similarity transform as the crop (rigid points: eye corners and
   nose tip -> canonical face), so head position, size and tilt are removed;
2. take the 40 lip points (20 outer, 20 inner) in that canonical face;
3. centre them on the outer-lip centre and divide by a fixed scale.

What is left is lip *shape*: opening, width, rounding, protrusion as seen from
the front. That is 80 numbers per frame instead of 7,744 pixels.
"""

from __future__ import annotations

import numpy as np

from punarvaani.preprocessing.lip_crop import (
    CANONICAL_STABLE,
    LIPS_INNER,
    LIPS_OUTER,
    STABLE_POINTS,
    apply_transform,
    similarity_transform,
)

LIP_POINTS = LIPS_OUTER + LIPS_INNER
FEATURE_DIM = 2 * len(LIP_POINTS)
_SCALE = 32.0  # canonical-face pixels; the mouth is ~50 px wide there, so features land in about [-1, 1]


def canonical_lips(landmarks: np.ndarray) -> np.ndarray:
    """(478, 2) face landmarks in image pixels -> (40, 2) lip points in the canonical face."""
    m = similarity_transform(landmarks[list(STABLE_POINTS)], CANONICAL_STABLE)
    return apply_transform(landmarks[list(LIP_POINTS)], m)


def lip_features(lips: np.ndarray) -> np.ndarray:
    """(..., 40, 2) canonical lip points -> (..., 80) shape features (centred, scaled)."""
    centre = lips[..., : len(LIPS_OUTER), :].mean(-2, keepdims=True)
    return ((lips - centre) / _SCALE).reshape(*lips.shape[:-2], FEATURE_DIM).astype(np.float32)


def with_deltas(seq: np.ndarray) -> np.ndarray:
    """(T, F) -> (T, 2F): shape plus its frame-to-frame change (how the lips are moving)."""
    delta = np.gradient(seq, axis=0) if len(seq) > 1 else np.zeros_like(seq)
    return np.concatenate([seq, delta], axis=-1)

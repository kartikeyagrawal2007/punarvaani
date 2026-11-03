"""
Pure-geometry helpers for mouth ROI cropping.

Kept free of MediaPipe so the box maths can be unit-tested without the
optional `mediapipe` dependency installed.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

# MediaPipe FaceMesh indices for the outer lip contour.
MOUTH_LANDMARK_INDICES: Tuple[int, ...] = (
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 308, 324, 318, 402, 317,
)

Box = Tuple[int, int, int, int]  # top, bottom, left, right (inclusive)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def landmark_pixels(
    landmarks: Sequence,
    indices: Iterable[int],
    image_width: int,
    image_height: int,
) -> list:
    """
    Convert normalised landmarks (``.x``/``.y`` or ``{"x","y"}``) into pixel points.
    """
    points = []
    for idx in indices:
        landmark = landmarks[idx]
        if isinstance(landmark, dict):
            x, y = landmark["x"], landmark["y"]
        else:
            x, y = landmark.x, landmark.y
        points.append((float(x) * image_width, float(y) * image_height))
    return points


def compute_mouth_box(
    points_xy: Sequence[Tuple[float, float]],
    image_width: int,
    image_height: int,
    prev_box: Optional[Box] = None,
    alpha: float = 0.6,
    margin_x: float = 0.35,
    margin_y: float = 0.55,
    square_scale: float = 1.3,
) -> Box:
    """
    Bounding box around lip points, expanded and squared, then temporally smoothed.

    ``alpha`` is the weight of the new observation (0..1); ``prev_box`` is the
    previous smoothed box. The result is clipped to the image.
    """
    if not points_xy:
        raise ValueError("points_xy must contain at least one point")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    xs = [p[0] for p in points_xy]
    ys = [p[1] for p in points_xy]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    cx = (max(xs) + min(xs)) / 2.0
    cy = (max(ys) + min(ys)) / 2.0

    expanded_w = width * (1.0 + 2.0 * margin_x)
    expanded_h = height * (1.0 + 2.0 * margin_y)
    side = max(1.0, square_scale * max(expanded_w, expanded_h))

    left = cx - side / 2.0
    right = cx + side / 2.0
    top = cy - side / 2.0
    bottom = cy + side / 2.0

    if prev_box is not None:
        p_top, p_bottom, p_left, p_right = prev_box
        top = alpha * top + (1.0 - alpha) * p_top
        bottom = alpha * bottom + (1.0 - alpha) * p_bottom
        left = alpha * left + (1.0 - alpha) * p_left
        right = alpha * right + (1.0 - alpha) * p_right

    return (
        _clamp(int(round(top)), 0, image_height - 1),
        _clamp(int(round(bottom)), 0, image_height - 1),
        _clamp(int(round(left)), 0, image_width - 1),
        _clamp(int(round(right)), 0, image_width - 1),
    )


def center_square_box(image_width: int, image_height: int, scale: float = 1.0) -> Box:
    """Largest centred square (optionally scaled) used as a detection fallback."""
    side = max(1, int(round(min(image_width, image_height) * max(0.0, scale))))
    left = max(0, (image_width - side) // 2)
    top = max(0, (image_height - side) // 2)
    return (top, min(image_height - 1, top + side - 1), left, min(image_width - 1, left + side - 1))

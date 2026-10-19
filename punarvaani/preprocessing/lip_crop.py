"""Stage [1]: face landmarks -> stabilised 96x96 grayscale mouth crops.

This reproduces the LRS3 / AV-HuBERT preprocessing recipe with MediaPipe Face
Mesh landmarks instead of dlib's 68 points:

1. detect landmarks on every frame, interpolate frames where the face was lost;
2. smooth landmarks over time (a centred moving average) so the crop does not
   jitter from frame to frame - jitter looks like lip motion to the model;
3. per frame, fit a similarity transform (rotation + uniform scale + shift)
   from a few *rigid* points (eye corners, nose tip) to a canonical face, so
   head pose and distance to the camera are normalised away;
4. warp the frame into that canonical 256x256 face and cut a 96x96 window
   centred on the (smoothed) mouth;
5. convert to grayscale. Training then takes a random 88x88 crop (centre crop
   at test time), which is the input size AV-HuBERT was pretrained on.

Mouth corners are deliberately *not* used for alignment: they move when the
person speaks, and aligning on them would normalise away the very motion we
want to read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# MediaPipe Face Mesh indices (478-point model, same for the legacy 468 subset).
LIPS_OUTER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
LIPS_INNER = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191)
MOUTH_CORNERS = (61, 291)
INNER_LIP_TOP, INNER_LIP_BOTTOM = 13, 14
# Rigid points: outer/inner eye corners and nose tip.
STABLE_POINTS = (33, 133, 362, 263, 1)
# Where those points sit in the canonical 256x256 face (eyes level, nose centred).
CANONICAL_SIZE = 256
CANONICAL_STABLE = np.array(
    [[78.0, 100.0], [110.0, 100.0], [146.0, 100.0], [178.0, 100.0], [128.0, 150.0]], dtype=np.float64
)


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (Umeyama 1991) mapping ``src`` onto ``dst``.

    Returns a 2x3 matrix ``M`` with ``dst ≈ src @ M[:, :2].T + M[:, 2]``.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    cov = d.T @ s / len(src)
    u, sig, vt = np.linalg.svd(cov)
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:  # avoid a reflection
        sign[1, 1] = -1
    rot = u @ sign @ vt
    var_s = (s**2).sum() / len(src)
    scale = np.trace(np.diag(sig) @ sign) / var_s
    m = np.zeros((2, 3))
    m[:, :2] = scale * rot
    m[:, 2] = mu_d - scale * rot @ mu_s
    return m


def apply_transform(points: np.ndarray, m: np.ndarray) -> np.ndarray:
    return points @ m[:, :2].T + m[:, 2]


def interpolate_missing(landmarks: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linearly interpolate (T, N, 2) landmarks over frames where ``valid`` is False."""
    if valid.all():
        return landmarks
    if not valid.any():
        raise ValueError("no face found in any frame")
    t = np.arange(len(landmarks))
    out = landmarks.copy()
    flat = out.reshape(len(out), -1)
    good = np.flatnonzero(valid)
    for j in range(flat.shape[1]):
        flat[:, j] = np.interp(t, good, flat[good, j])
    return out


def smooth_landmarks(landmarks: np.ndarray, window: int = 12) -> np.ndarray:
    """Centred moving average over time with edge padding (window 12 = 0.48 s at 25 fps)."""
    if window <= 1 or len(landmarks) < 2:
        return landmarks
    left, right = window // 2, window - 1 - window // 2
    padded = np.concatenate(
        [np.repeat(landmarks[:1], left, 0), landmarks, np.repeat(landmarks[-1:], right, 0)]
    )
    csum = np.cumsum(padded, axis=0, dtype=np.float64)
    csum = np.concatenate([np.zeros_like(csum[:1]), csum])
    return ((csum[window:] - csum[:-window]) / window).astype(landmarks.dtype)


def _warp_affine(img: np.ndarray, m: np.ndarray, size: int) -> np.ndarray:
    try:
        import cv2

        return cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    except ImportError:  # pure-numpy bilinear fallback (slow, used in tests without OpenCV)
        inv = np.linalg.inv(np.vstack([m, [0, 0, 1]]))[:2]
        ys, xs = np.mgrid[0:size, 0:size]
        src = apply_transform(np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64), inv)
        x = np.clip(src[:, 0], 0, img.shape[1] - 1.001)
        y = np.clip(src[:, 1], 0, img.shape[0] - 1.001)
        x0, y0 = x.astype(int), y.astype(int)
        fx, fy = x - x0, y - y0
        if img.ndim == 3:
            fx, fy = fx[:, None], fy[:, None]
        v = (
            img[y0, x0] * (1 - fx) * (1 - fy)
            + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy
            + img[y0 + 1, x0 + 1] * fx * fy
        )
        return v.reshape(size, size, *img.shape[2:]).astype(img.dtype)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    # ITU-R BT.601 luma, the same weights OpenCV uses (input assumed RGB).
    return (img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114).astype(img.dtype)


@dataclass
class LipCropResult:
    crops: np.ndarray  # (T, crop, crop) uint8
    mouth_landmarks: np.ndarray  # (T, 40, 2) lip points in canonical-face coordinates
    missing_ratio: float


class LipCropper:
    def __init__(self, crop_size: int = 96, smooth_window: int = 12, max_missing_ratio: float = 0.1):
        self.crop_size = crop_size
        self.smooth_window = smooth_window
        self.max_missing_ratio = max_missing_ratio

    def __call__(
        self, frames: np.ndarray, landmarks: np.ndarray, valid: np.ndarray | None = None
    ) -> LipCropResult:
        """``frames``: (T, H, W[, 3]) uint8; ``landmarks``: (T, N, 2) pixel coordinates."""
        if valid is None:
            valid = np.ones(len(frames), dtype=bool)
        missing = float((~valid).mean())
        if missing > self.max_missing_ratio:
            raise ValueError(f"face missing in {missing:.0%} of frames (limit {self.max_missing_ratio:.0%})")
        lms = smooth_landmarks(interpolate_missing(landmarks.astype(np.float64), valid), self.smooth_window)

        transforms = [similarity_transform(lm[list(STABLE_POINTS)], CANONICAL_STABLE) for lm in lms]
        lips_idx = list(LIPS_OUTER + LIPS_INNER)
        mouth = np.stack([apply_transform(lm[lips_idx], m) for lm, m in zip(lms, transforms)])
        centres = smooth_landmarks(mouth.mean(1, keepdims=True), self.smooth_window)[:, 0]

        half = self.crop_size / 2
        crops = np.empty((len(frames), self.crop_size, self.crop_size), dtype=np.uint8)
        for t, (frame, m, c) in enumerate(zip(frames, transforms, centres)):
            # Fold "shift the mouth centre to the crop centre" into the warp so we
            # warp once, straight to the crop, instead of warping 256x256 and slicing.
            shifted = m.copy()
            shifted[:, 2] += half - c
            crops[t] = to_gray(_warp_affine(frame, shifted, self.crop_size))
        return LipCropResult(crops=crops, mouth_landmarks=mouth, missing_ratio=float(missing))


def mouth_opening(mouth_landmarks: np.ndarray) -> np.ndarray:
    """Inner-lip gap divided by mouth width, per frame (scale invariant, ~0 when closed).

    ``mouth_landmarks`` is the (T, 40, 2) array from :class:`LipCropResult`
    (20 outer then 20 inner points). Used for lip-activity detection.
    """
    outer = mouth_landmarks[:, : len(LIPS_OUTER)]
    inner = mouth_landmarks[:, len(LIPS_OUTER) :]
    width = np.linalg.norm(outer[:, LIPS_OUTER.index(61)] - outer[:, LIPS_OUTER.index(291)], axis=-1)
    gap = np.linalg.norm(
        inner[:, LIPS_INNER.index(INNER_LIP_TOP)] - inner[:, LIPS_INNER.index(INNER_LIP_BOTTOM)], axis=-1
    )
    return gap / np.maximum(width, 1e-6)


class MediaPipeLandmarker:
    """Face landmarks with MediaPipe Tasks (``FaceLandmarker``) in video mode.

    Download the model once:
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
    """

    def __init__(self, model_path: str | Path, fps: float = 25.0):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._frame_ms = 1000.0 / fps
        self._t = 0

    def __call__(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """Returns (478, 2) pixel coordinates, or None if no face was found."""
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        result = self._landmarker.detect_for_video(image, int(self._t * self._frame_ms))
        self._t += 1
        if not result.face_landmarks:
            return None
        h, w = frame_rgb.shape[:2]
        return np.array([[p.x * w, p.y * h] for p in result.face_landmarks[0]], dtype=np.float64)

    def detect_all(self, frames_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lms, valid = [], []
        for frame in frames_rgb:
            lm = self(frame)
            valid.append(lm is not None)
            lms.append(lm if lm is not None else np.zeros((478, 2)))
        return np.stack(lms), np.array(valid)

    def close(self) -> None:
        self._landmarker.close()


def read_video_rgb(path: str | Path, target_fps: float = 25.0) -> tuple[np.ndarray, float]:
    """Read a video with OpenCV, resampled to ``target_fps`` by frame dropping/repeating."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"could not read any frame from {path}")
    video = np.stack(frames)
    if abs(fps - target_fps) > 0.01:
        idx = np.round(np.arange(0, len(video), fps / target_fps)).astype(int)
        video = video[np.clip(idx, 0, len(video) - 1)]
    return video, target_fps


def crop_video_file(path: str | Path, landmarker: MediaPipeLandmarker, cropper: LipCropper | None = None):
    frames, _ = read_video_rgb(path)
    lms, valid = landmarker.detect_all(frames)
    return (cropper or LipCropper())(frames, lms, valid)

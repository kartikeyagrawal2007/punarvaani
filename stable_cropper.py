"""
Stable mouth crops for whole clips (the standard lip-reading preprocessing).

The per-frame FaceCropper sizes each crop from the lip HEIGHT, so the crop zooms out whenever the mouth opens
(hiding exactly what the model must see), and it smooths the box in the coordinates of a face box that moves
every frame. This cropper instead:

1. finds FaceMesh landmarks in every frame and maps them to full-frame pixels;
2. sizes the crop from the distance between the eyes, which follows camera zoom but never the lips;
3. smooths the mouth centre, crop size and head roll with a centred moving average;
4. cuts every frame with one affine warp (rotate upright, scale, centre the mouth).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import cv2
import numpy as np

from mouth_geometry import (
    EYE_CORNERS,
    MOUTH_LANDMARK_INDICES,
    crop_affine,
    stable_crop_track,
)


class StableMouthCropper:
    def __init__(self, window: int = 9, side_scale: float = 1.2, min_confidence: float = 0.5,
                 face_margin: float = 0.4):
        import mediapipe as mp

        # Faces in news video are too small for FaceMesh's own detector, so (like FaceCropper) we find the
        # face with the full-range detector first and run FaceMesh on that region; unlike FaceCropper, the
        # landmarks are then mapped back to full-frame pixels, so smoothing works in ONE coordinate system.
        self._detector_factory = lambda: mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=min_confidence
        )
        self._mesh_factory = lambda: mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=min_confidence,
        )
        self.window = window
        self.side_scale = side_scale
        self.face_margin = face_margin

    def _landmarks(self, frame, detector, mesh):
        """FaceMesh landmarks of the largest face as an (468, 2) array in full-frame pixels, or None."""
        h, w = frame.shape[:2]
        detections = detector.process(frame).detections or []
        boxes = [d.location_data.relative_bounding_box for d in detections]
        boxes = [b for b in boxes if b.width > 0 and b.height > 0]
        if not boxes:
            return None
        b = max(boxes, key=lambda b: b.width * b.height)
        side = max(b.width * w, b.height * h) * (1.0 + 2.0 * self.face_margin)
        cx, cy = (b.xmin + b.width / 2) * w, (b.ymin + b.height / 2) * h
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x1, y1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        region = np.ascontiguousarray(frame[y0:y1, x0:x1])
        result = mesh.process(region)
        if not result.multi_face_landmarks:
            return None
        lms = result.multi_face_landmarks[0].landmark
        return np.array([[lm.x * (x1 - x0) + x0, lm.y * (y1 - y0) + y0] for lm in lms])

    @staticmethod
    def _measure(points: np.ndarray):
        lips = points[list(MOUTH_LANDMARK_INDICES)]
        center = (lips.min(axis=0) + lips.max(axis=0)) / 2.0
        eye_l, eye_r = points[EYE_CORNERS[0]], points[EYE_CORNERS[1]]
        eye_distance = float(np.linalg.norm(eye_r - eye_l))
        roll = math.degrees(math.atan2(eye_r[1] - eye_l[1], eye_r[0] - eye_l[0]))
        return center, eye_distance, roll

    def crop_clip(self, frames_rgb: Sequence[np.ndarray], frame_size: int = 224) -> Optional[List[np.ndarray]]:
        """RGB frames -> RGB mouth crops (frame_size x frame_size), or None when the face is lost too often."""
        if not len(frames_rgb):
            return None
        centers, widths, rolls, valid = [], [], [], []
        detector, mesh = self._detector_factory(), self._mesh_factory()
        try:
            for frame in frames_rgb:
                points = self._landmarks(frame, detector, mesh)
                if points is not None:
                    c, mw, r = self._measure(points)
                    centers.append(c)
                    widths.append(mw)
                    rolls.append(r)
                    valid.append(True)
                else:
                    centers.append((0.0, 0.0))
                    widths.append(0.0)
                    rolls.append(0.0)
                    valid.append(False)
        finally:
            detector.close()
            mesh.close()

        track = stable_crop_track(centers, widths, rolls, valid, window=self.window, side_scale=self.side_scale)
        if track is None:
            return None
        cx, cy, roll, side = track
        return [
            cv2.warpAffine(frame, crop_affine(cx[i], cy[i], roll[i], side[i], frame_size), (frame_size, frame_size),
                           flags=cv2.INTER_AREA if side[i] > frame_size else cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
            for i, frame in enumerate(frames_rgb)
        ]

"""Shot detection and face tracking for mined videos.

A news video cuts between anchor, reporter and B-roll every few seconds. A clip
must contain *one continuous shot* of *one face* that is large enough to read
lips from, so:

1. :func:`shot_boundaries` - colour-histogram difference between consecutive
   frames; a spike means a cut;
2. :class:`IoUTracker` - links face boxes across frames by overlap (the face
   barely moves between 40 ms frames), producing tracks;
3. :func:`usable_tracks` - keeps tracks that are long enough, have a big enough
   face (mouth crops from tiny faces are mush) and do not share the frame with
   another large face (who is talking?).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


def shot_boundaries(frames: np.ndarray, bins: int = 16, threshold: float = 0.4) -> list[int]:
    """Frame indices where a new shot starts (always includes 0). ``frames``: (T, H, W, 3) uint8."""
    hists = []
    for f in frames:
        h = np.concatenate(
            [np.histogram(f[..., c], bins=bins, range=(0, 256))[0] for c in range(f.shape[-1])]
        )
        hists.append(h / max(1, h.sum()))
    cuts = [0]
    for t in range(1, len(hists)):
        if 0.5 * np.abs(hists[t] - hists[t - 1]).sum() > threshold:  # total-variation distance
            cuts.append(t)
    return cuts


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
    return inter / max(1e-9, area(a) + area(b) - inter)


@dataclass
class Track:
    track_id: int
    frames: list[int] = field(default_factory=list)
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.frames)

    def median_size(self) -> float:
        return float(np.median([b[3] - b[1] for b in self.boxes])) if self.boxes else 0.0


class IoUTracker:
    def __init__(self, min_iou: float = 0.4, max_gap: int = 3):
        self.min_iou, self.max_gap = min_iou, max_gap

    def __call__(self, detections: Sequence[Sequence[tuple[float, float, float, float]]],
                 shot_starts: Sequence[int] = (0,)) -> list[Track]:  # fmt: skip
        """``detections[t]`` = face boxes (x1, y1, x2, y2) in frame t. Tracks never cross a shot cut."""
        finished: list[Track] = []
        active: list[Track] = []
        next_id = 0
        cuts = set(shot_starts)
        for t, boxes in enumerate(detections):
            if t in cuts and t > 0:
                finished += active
                active = []
            unmatched = list(range(len(boxes)))
            for tr in sorted(active, key=lambda tr: -tr.length):
                if not unmatched:
                    break
                scores = [iou(tr.boxes[-1], boxes[j]) for j in unmatched]
                k = int(np.argmax(scores))
                if scores[k] >= self.min_iou:
                    j = unmatched.pop(k)
                    tr.frames.append(t)
                    tr.boxes.append(tuple(boxes[j]))
            still = []
            for tr in active:
                (still if t - tr.frames[-1] <= self.max_gap else finished).append(tr)
            active = still
            for j in unmatched:
                active.append(Track(next_id, [t], [tuple(boxes[j])]))
                next_id += 1
        return sorted(finished + active, key=lambda tr: tr.track_id)


def usable_tracks(tracks: Sequence[Track], frame_height: int, min_frames: int = 25, min_face_frac: float = 0.15,
                  competing_face_frac: float = 0.10) -> list[Track]:  # fmt: skip
    keep = []
    for tr in tracks:
        if tr.length < min_frames or tr.median_size() < min_face_frac * frame_height:
            continue
        span = set(tr.frames)
        rivals = [
            o
            for o in tracks
            if o is not tr and span & set(o.frames) and o.median_size() >= competing_face_frac * frame_height
        ]
        if not rivals:
            keep.append(tr)
    return keep

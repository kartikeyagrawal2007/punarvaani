"""
Synthetic Hindi VSR dataset generator.

Creates the same on-disk contract as the real handover:

    <output_dir>/spk01/spk01_001.mp4
    <output_dir>/spk01/spk01_001.txt

The clips contain no real faces. Use them with ``crop_mode="none"`` for
plumbing/smoke tests (dataset, collate, model, training, evaluation).
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

DEFAULT_TEXTS: Sequence[str] = (
    "नमस्ते भारत",
    "मुझे पानी चाहिए",
    "डॉक्टर को बुलाओ",
    "मैं ठीक हूँ",
    "दर्द हो रहा है",
    "यह कितना है",
)


def _write_clip(
    path: str,
    num_frames: int,
    fps: int,
    size: Tuple[int, int],
    seed: int,
) -> None:
    height, width = size
    rng = np.random.RandomState(seed)
    base = rng.randint(40, 180, size=3).tolist()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")

    try:
        for t in range(num_frames):
            frame = np.full((height, width, 3), base, dtype=np.uint8)
            cx = int((t / max(1, num_frames - 1)) * (width - 1))
            color = ((base[0] + t * 3) % 256, (base[1] + t * 5) % 256, (base[2] + t * 7) % 256)
            cv2.circle(frame, (max(1, cx), height // 2), max(2, width // 8), color, -1)
            writer.write(frame)
    finally:
        writer.release()


def create_synthetic_dataset(
    output_dir: str,
    speakers: int = 3,
    clips: int = 2,
    seconds: float = 1.5,
    fps: int = 25,
    size: Tuple[int, int] = (96, 96),
    texts: Sequence[str] = DEFAULT_TEXTS,
    seed: int = 0,
) -> Dict[str, List[dict]]:
    """
    Write ``speakers * clips`` video/transcript pairs into speaker subfolders.

    Returns a manifest dict: ``{"samples": [...], "speakers": {...}}``.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_frames = max(4, int(round(seconds * fps)))
    samples: List[dict] = []
    speaker_names = [f"spk{i + 1:02d}" for i in range(speakers)]

    for s_idx, speaker in enumerate(speaker_names):
        speaker_dir = os.path.join(output_dir, speaker)
        os.makedirs(speaker_dir, exist_ok=True)
        for c_idx in range(clips):
            base = f"{speaker}_{c_idx + 1:03d}"
            video_path = os.path.join(speaker_dir, f"{base}.mp4")
            text_path = os.path.join(speaker_dir, f"{base}.txt")
            _write_clip(video_path, num_frames, fps, size, seed + s_idx * 100 + c_idx)
            text = texts[(s_idx * clips + c_idx) % len(texts)]
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            samples.append(
                {
                    "video": video_path,
                    "text": text_path,
                    "speaker": speaker,
                    "num_frames": num_frames,
                    "transcript": text,
                }
            )

    return {"samples": samples, "speakers": {s: clips for s in speaker_names}}

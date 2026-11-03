"""
Offline mouth preprocessing for Punarvaani.

Walks an input tree containing paired videos/transcripts, crops a stabilised
mouth ROI with MediaPipe, and writes the same relative tree under ``--output_dir``
so speaker subfolders are preserved.

Output contract matches the training loader:
    <output_dir>/<speaker>/<clip>.mp4
    <output_dir>/<speaker>/<clip>.txt
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

try:
    from face_cropper import FaceCropper
    HAS_FACE_CROPPER = True
except ImportError:
    HAS_FACE_CROPPER = False


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def _select_indices(total: int, num_frames: int, max_frames: int, frame_stride: int) -> np.ndarray:
    if total <= 0:
        return np.zeros((0,), dtype=np.int64)
    if num_frames > 0:
        count = min(num_frames, total)
        if count >= total:
            return np.arange(total, dtype=np.int64)
        return np.linspace(0, total - 1, count, dtype=np.int64)
    indices = np.arange(0, total, max(1, frame_stride), dtype=np.int64)
    return indices[:max_frames]


def preprocess_dataset(
    input_dir: str,
    output_dir: str,
    num_frames: int = 0,
    frame_size: int = 224,
    max_frames: int = 96,
    frame_stride: int = 1,
    fps: float = 0.0,
    cropper=None,
) -> dict:
    """Crop mouth ROIs for every video/transcript pair under ``input_dir``."""
    if cropper is None:
        if not HAS_FACE_CROPPER:
            raise RuntimeError(
                "MediaPipe FaceCropper is unavailable. Install mediapipe or pass a cropper."
            )
        cropper = FaceCropper(
            min_face_detector_confidence=0.5,
            face_detector_model_selection="SHORT_RANGE",
            landmark_detector_static_image_mode="STATIC_MODE",
            min_landmark_detector_confidence=0.5,
        )

    os.makedirs(output_dir, exist_ok=True)

    video_files = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.lower().endswith(VIDEO_EXTENSIONS):
                video_files.append(os.path.join(root, name))
    video_files.sort()

    if not video_files:
        print(f"[Warning] No video files found in '{input_dir}'.")
        return {
            "total_videos": 0,
            "successful_crops": 0,
            "failed_videos": [],
            "crop_success_rate_pct": 0.0,
        }

    report = {
        "total_videos": len(video_files),
        "successful_crops": 0,
        "failed_videos": [],
        "crop_success_rate_pct": 0.0,
    }

    print(f"Found {len(video_files)} video files in '{input_dir}'. Processing...")

    for video_path in tqdm(video_files, desc="Preprocessing"):
        relative = os.path.relpath(video_path, input_dir)
        relative_dir = os.path.dirname(relative)
        base_name = os.path.splitext(os.path.basename(relative))[0]
        target_dir = os.path.join(output_dir, relative_dir) if relative_dir else output_dir
        os.makedirs(target_dir, exist_ok=True)
        out_video = os.path.join(target_dir, f"{base_name}.mp4")

        # Copy the matching transcript into the same relative location.
        txt_path = os.path.splitext(video_path)[0] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as src:
                text = src.read()
            with open(os.path.join(target_dir, f"{base_name}.txt"), "w", encoding="utf-8") as dst:
                dst.write(text)

        cap = cv2.VideoCapture(video_path)
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = []
        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            cap.release()

        output_fps = fps if fps > 0 else (source_fps if source_fps > 0 else 25.0)

        if not frames:
            report["failed_videos"].append({"video": video_path, "reason": "unreadable video"})
            continue

        indices = _select_indices(len(frames), num_frames, max_frames, frame_stride)
        if len(indices) == 0:
            report["failed_videos"].append({"video": video_path, "reason": "no frames selected"})
            continue

        if hasattr(cropper, "reset"):
            cropper.reset()

        cropped = []
        failure = None
        for idx in indices:
            rgb = cv2.cvtColor(frames[int(idx)], cv2.COLOR_BGR2RGB)
            crop = cropper.crop_mouth(rgb, frame_size=(frame_size, frame_size))
            if crop is None or getattr(crop, "size", 0) == 0:
                failure = "mouth crop failed"
                break
            cropped.append(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        if failure is not None or not cropped:
            report["failed_videos"].append({"video": video_path, "reason": failure or "no crops"})
            continue

        writer = cv2.VideoWriter(
            out_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(output_fps),
            (frame_size, frame_size),
        )
        if not writer.isOpened():
            report["failed_videos"].append({"video": video_path, "reason": "writer failed"})
            continue
        try:
            for frame in cropped:
                writer.write(frame)
        finally:
            writer.release()

        report["successful_crops"] += 1

    report["crop_success_rate_pct"] = (
        report["successful_crops"] / report["total_videos"] * 100.0
        if report["total_videos"]
        else 0.0
    )

    report_path = os.path.join(output_dir, "preprocessing_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\nPreprocessing Complete:")
    print(
        f"  Success: {report['successful_crops']}/{report['total_videos']} "
        f"({report['crop_success_rate_pct']:.1f}%)"
    )
    print(f"  Diagnostic Report saved to: {report_path}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Punarvaani Offline Lip Preprocessing")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for cropped videos")
    parser.add_argument("--num_frames", type=int, default=0, help="Uniform frames per clip; 0 = all up to --max_frames")
    parser.add_argument("--max_frames", type=int, default=96, help="Maximum frames per clip when --num_frames 0")
    parser.add_argument("--frame_stride", type=int, default=1, help="Read every Nth frame")
    parser.add_argument("--frame_size", type=int, default=224, help="Output resolution (H=W)")
    parser.add_argument("--fps", type=float, default=0.0, help="Output frame rate; 0 = preserve source FPS")
    args = parser.parse_args(argv)

    try:
        report = preprocess_dataset(
            args.input_dir,
            args.output_dir,
            num_frames=args.num_frames,
            frame_size=args.frame_size,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            fps=args.fps,
        )
    except RuntimeError as exc:
        print(f"[Error] {exc}")
        return 1
    return 0 if report["total_videos"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

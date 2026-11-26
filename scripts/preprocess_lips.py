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
from functools import partial

os.environ.setdefault("GLOG_minloglevel", "2")  # hide MediaPipe/TFLite C++ warnings (one per model load)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from face_cropper import FaceCropper  # noqa: E402
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


def _make_cropper(crop_method: str):
    if crop_method == "stable":
        try:
            from stable_cropper import StableMouthCropper
        except ImportError as exc:
            raise RuntimeError(f"The stable cropper needs mediapipe: {exc}") from exc
        return StableMouthCropper()
    if not HAS_FACE_CROPPER:
        raise RuntimeError("MediaPipe FaceCropper is unavailable. Install mediapipe or pass a cropper.")
    return FaceCropper(
        min_face_detector_confidence=0.5,
        face_detector_model_selection="SHORT_RANGE",
        landmark_detector_static_image_mode="STATIC_MODE",
        min_landmark_detector_confidence=0.5,
    )


def _process_video(video_path, input_dir, output_dir, num_frames, frame_size, max_frames, frame_stride, fps,
                   skip_existing, cropper):
    """Crop one clip. Returns (video_path, None) on success or (video_path, reason) on failure."""
    relative = os.path.relpath(video_path, input_dir)
    relative_dir = os.path.dirname(relative)
    base_name = os.path.splitext(os.path.basename(relative))[0]
    target_dir = os.path.join(output_dir, relative_dir) if relative_dir else output_dir
    os.makedirs(target_dir, exist_ok=True)
    out_video = os.path.join(target_dir, f"{base_name}.mp4")
    if skip_existing and os.path.exists(out_video) and os.path.getsize(out_video) > 0:
        return video_path, None

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
        return video_path, "unreadable video"
    indices = _select_indices(len(frames), num_frames, max_frames, frame_stride)
    if len(indices) == 0:
        return video_path, "no frames selected"

    cropped = []
    if hasattr(cropper, "crop_clip"):  # whole-clip cropper: sees every frame before cutting any
        rgb_frames = [cv2.cvtColor(frames[int(idx)], cv2.COLOR_BGR2RGB) for idx in indices]
        crops = cropper.crop_clip(rgb_frames, frame_size)
        if crops is None:
            return video_path, "face not found in enough frames"
        cropped = [cv2.cvtColor(crop, cv2.COLOR_RGB2BGR) for crop in crops]
    else:
        if hasattr(cropper, "reset"):
            cropper.reset()
        for idx in indices:
            rgb = cv2.cvtColor(frames[int(idx)], cv2.COLOR_BGR2RGB)
            crop = cropper.crop_mouth(rgb, frame_size=(frame_size, frame_size))
            if crop is None or getattr(crop, "size", 0) == 0:
                return video_path, "mouth crop failed"
            cropped.append(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    if not cropped:
        return video_path, "no crops"

    # Write to a temporary name and rename at the end, so an interrupted run never leaves a
    # half-written clip that --skip_existing would mistake for a finished one.
    part_video = os.path.join(target_dir, f"{base_name}.part.mp4")
    writer = cv2.VideoWriter(part_video, cv2.VideoWriter_fourcc(*"mp4v"), float(output_fps), (frame_size, frame_size))
    if not writer.isOpened():
        return video_path, "writer failed"
    try:
        for frame in cropped:
            writer.write(frame)
    finally:
        writer.release()
    os.replace(part_video, out_video)
    return video_path, None


_WORKER_CROPPER = None


def _init_worker(crop_method):
    global _WORKER_CROPPER
    _WORKER_CROPPER = _make_cropper(crop_method)  # loaded once per worker, reused for every clip


def _worker_task(video_path, **kwargs):
    try:
        return _process_video(video_path, cropper=_WORKER_CROPPER, **kwargs)
    except Exception as exc:  # one bad clip must not stop a long run
        return video_path, f"error: {exc}"


def preprocess_dataset(
    input_dir: str,
    output_dir: str,
    num_frames: int = 0,
    frame_size: int = 224,
    max_frames: int = 96,
    frame_stride: int = 1,
    fps: float = 0.0,
    cropper=None,
    crop_method: str = "stable",
    skip_existing: bool = False,
    workers: int = 1,
) -> dict:
    """Crop mouth ROIs for every video/transcript pair under ``input_dir``.

    ``crop_method="stable"`` (default) crops whole clips with StableMouthCropper: one coordinate system,
    crop size from the eye distance, smoothed over time. ``"legacy"`` is the old per-frame FaceCropper,
    whose crop zooms with the lip height. ``workers`` > 1 crops clips in parallel processes (each loads its
    own models once); a cropper passed in always runs in this process."""
    if cropper is None and workers <= 1:
        cropper = _make_cropper(crop_method)

    os.makedirs(output_dir, exist_ok=True)

    video_files = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.lower().endswith(VIDEO_EXTENSIONS) and ".part." not in name:
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
    settings = dict(input_dir=input_dir, output_dir=output_dir, num_frames=num_frames, frame_size=frame_size,
                    max_frames=max_frames, frame_stride=frame_stride, fps=fps, skip_existing=skip_existing)

    if cropper is not None:
        results = (_process_video(v, cropper=cropper, **settings) for v in video_files)
        pool = None
    else:
        import multiprocessing as mp

        pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(crop_method,))
        results = pool.imap_unordered(partial(_worker_task, **settings), video_files, chunksize=4)
    try:
        for video_path, failure in tqdm(results, total=len(video_files), desc="Preprocessing"):
            if failure is None:
                report["successful_crops"] += 1
            else:
                report["failed_videos"].append({"video": video_path, "reason": failure})
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

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
    parser.add_argument("--crop_method", choices=["stable", "legacy"], default="stable",
                        help="stable: steady whole-clip crops sized by eye distance (default); "
                             "legacy: the old per-frame crops that zoom with the lip height")
    parser.add_argument("--workers", type=int, default=0,
                        help="Clips cropped in parallel; 0 = half the CPU cores (at most 8)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip clips whose cropped .mp4 already exists (to continue an interrupted run)")
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
            crop_method=args.crop_method,
            skip_existing=args.skip_existing,
            workers=args.workers if args.workers > 0 else max(1, min(8, (os.cpu_count() or 2) // 2)),
        )
    except RuntimeError as exc:
        print(f"[Error] {exc}")
        return 1
    return 0 if report["total_videos"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

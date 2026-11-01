"""
Offline Mouth Preprocessing Script for Punarvaani
Extracts and stabilizes 224x224 mouth Region of Interest (ROI) from video files using MediaPipe.
Saves preprocessed video frames and logs a comprehensive diagnostic report.
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm

try:
    from face_cropper import FaceCropper
    HAS_FACE_CROPPER = True
except ImportError:
    HAS_FACE_CROPPER = False


def preprocess_dataset(
    input_dir: str,
    output_dir: str,
    num_frames: int = 16,
    frame_size: int = 224,
    smoothing_alpha: float = 0.7
):
    """
    Scans input_dir for video files, extracts stabilized mouth ROIs, and saves them to output_dir.
    """
    if not HAS_FACE_CROPPER:
        print("[Error] MediaPipe FaceCropper is not available in the environment. Please install mediapipe.")
        return

    os.makedirs(output_dir, exist_ok=True)
    cropper = FaceCropper(
        min_face_detector_confidence=0.5,
        face_detector_model_selection="SHORT_RANGE",
        landmark_detector_static_image_mode="STATIC_MODE",
        min_landmark_detector_confidence=0.5
    )

    valid_exts = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(valid_exts):
                video_files.append(os.path.join(root, f))

    if not video_files:
        print(f"[Warning] No video files found in '{input_dir}'.")
        return

    report = {
        "total_videos": len(video_files),
        "successful_crops": 0,
        "failed_videos": [],
        "crop_success_rate_pct": 0.0
    }

    print(f"Found {len(video_files)} video files in '{input_dir}'. Processing...")

    for vpath in tqdm(video_files, desc="Preprocessing"):
        base_name = os.path.splitext(os.path.basename(vpath))[0]
        out_vpath = os.path.join(output_dir, f"{base_name}.mp4")

        # Copy matching transcript if exists
        txt_path = os.path.splitext(vpath)[0] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as src_txt:
                with open(os.path.join(output_dir, f"{base_name}.txt"), 'w', encoding='utf-8') as dst_txt:
                    dst_txt.write(src_txt.read())

        cap = cv2.VideoCapture(vpath)
        raw_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(frame)
        cap.release()

        if len(raw_frames) < num_frames:
            report["failed_videos"].append({"video": vpath, "reason": f"Insufficient frames: {len(raw_frames)} < {num_frames}"})
            continue

        # Sample num_frames evenly
        indices = np.linspace(0, len(raw_frames) - 1, num_frames, dtype=int)
        sampled_frames = [raw_frames[i] for i in indices]

        # Crop mouth regions
        cropped_frames = []
        detection_failed = False

        for frame in sampled_frames:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mouth_crop = cropper.crop_face(rgb_frame)
                if mouth_crop is None:
                    detection_failed = True
                    break
                mouth_crop_bgr = cv2.cvtColor(mouth_crop, cv2.COLOR_RGB2BGR)
                mouth_crop_resized = cv2.resize(mouth_crop_bgr, (frame_size, frame_size))
                cropped_frames.append(mouth_crop_resized)
            except Exception as e:
                detection_failed = True
                break

        if detection_failed or len(cropped_frames) < num_frames:
            report["failed_videos"].append({"video": vpath, "reason": "Face/mouth landmark detection failed"})
            continue

        # Save cropped video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_vpath, fourcc, 25.0, (frame_size, frame_size))
        for cf in cropped_frames:
            out_writer.write(cf)
        out_writer.release()

        report["successful_crops"] += 1

    report["crop_success_rate_pct"] = (report["successful_crops"] / report["total_videos"] * 100) if report["total_videos"] > 0 else 0.0

    report_path = os.path.join(output_dir, "preprocessing_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"\nPreprocessing Complete:")
    print(f"  Success: {report['successful_crops']}/{report['total_videos']} ({report['crop_success_rate_pct']:.1f}%)")
    print(f"  Diagnostic Report saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Punarvaani Offline Lip Preprocessing")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw video files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save preprocessed cropped videos")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to sample per video")
    parser.add_argument("--frame_size", type=int, default=224, help="Output resolution (H=W)")
    args = parser.parse_args()

    preprocess_dataset(args.input_dir, args.output_dir, args.num_frames, args.frame_size)


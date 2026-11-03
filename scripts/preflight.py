"""
Preflight checks for a Punarvaani handover.

Validates the on-disk dataset and the Sarvam-1 LoRA adapter before any GPU time
is spent. Exits non-zero when a blocking issue is found.

Usage:
    python scripts/preflight.py --data_dir hindi_dataset \
        --sarvam_adapter_path checkpoints/sarvam_lora
"""

import argparse
import json
import math
import os
import re
import sys
from typing import Optional

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data.hindi_g2p import (  # noqa: E402
    clear_unsupported_char_log,
    devanagari_to_phonemes,
    get_unsupported_char_report,
)
from config import get_hindi_vocab  # noqa: E402

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
NON_DEVANAGARI_RE = re.compile(r"[^\u0900-\u097F\s।॥.,!?;:\"\'\-]")


def _probe_video(path: str):
    cap = cv2.VideoCapture(path)
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    return frames, fps


def _ctc_min(ids):
    if not ids:
        return 0
    repeats = sum(1 for a, b in zip(ids, ids[1:]) if a == b)
    return len(ids) + repeats


def scan_dataset(
    data_dir: str,
    max_frames: int = 96,
    tubelet_size: int = 2,
    temporal_upsample_factor: int = 4,
    max_invalid_sample_pct: float = 10.0,
    allow_unknown_speakers: bool = False,
) -> dict:
    vocab = get_hindi_vocab()
    clear_unsupported_char_log()

    videos = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            if name.lower().endswith(VIDEO_EXTENSIONS):
                videos.append(os.path.join(root, name))
    videos.sort()

    report = {
        "data_dir": os.path.abspath(data_dir),
        "videos": len(videos),
        "pairs": 0,
        "missing_transcripts": [],
        "orphan_transcripts": [],
        "unknown_speaker": [],
        "empty_transcripts": [],
        "non_devanagari": [],
        "speakers": {},
        "durations_sec": [],
        "phoneme_counts": [],
        "invalid_ctc": [],
        "invalid_pct": 0.0,
        "unsupported_g2p_chars": {},
        "blocking": [],
        "warnings": [],
    }

    transcript_seen = set()
    blocking = report["blocking"]
    warnings = report["warnings"]

    if not os.path.isdir(data_dir):
        blocking.append(f"data_dir does not exist: {data_dir}")
        return report

    for video in videos:
        rel = os.path.relpath(video, data_dir)
        parts = os.path.normpath(rel).split(os.sep)
        speaker = parts[0] if len(parts) > 1 else None
        base = os.path.splitext(video)[0]
        txt = base + ".txt"
        transcript_seen.add(os.path.abspath(txt))

        if speaker is None:
            report["unknown_speaker"].append(rel)
        else:
            report["speakers"].setdefault(speaker, {"clips": 0, "frames": 0})
            report["speakers"][speaker]["clips"] += 1

        if not os.path.exists(txt):
            report["missing_transcripts"].append(rel)
            continue

        with open(txt, "r", encoding="utf-8") as handle:
            text = " ".join(line.strip() for line in handle if line.strip())

        if not text.strip():
            report["empty_transcripts"].append(rel)
            continue

        if not DEVANAGARI_RE.search(text):
            report["non_devanagari"].append(rel)
        else:
            bad = NON_DEVANAGARI_RE.findall(text)
            if bad:
                report["non_devanagari"].append({"file": rel, "chars": sorted(set(bad))})

        phonemes = devanagari_to_phonemes(text, transcript_id=rel)
        ids = [vocab[p] for p in phonemes if p in vocab]
        ctc_min = _ctc_min(ids)
        report["pairs"] += 1
        report["phoneme_counts"].append(len(ids))

        frames, fps = _probe_video(video)
        if frames > 0 and fps > 0:
            report["durations_sec"].append(frames / fps)
        used = min(frames, max_frames) if frames > 0 else max_frames
        t_ctc = int(math.ceil(used / tubelet_size) * temporal_upsample_factor)
        if ctc_min > t_ctc:
            report["invalid_ctc"].append(
                {
                    "file": rel,
                    "phoneme_steps": ctc_min,
                    "t_ctc": t_ctc,
                    "frames": frames,
                }
            )
        if speaker is not None and frames > 0:
            report["speakers"][speaker]["frames"] += frames

    # Orphan transcripts (txt without a matching video)
    for root, _, files in os.walk(data_dir):
        for name in files:
            if not name.lower().endswith(".txt"):
                continue
            path = os.path.abspath(os.path.join(root, name))
            if path not in transcript_seen:
                report["orphan_transcripts"].append(os.path.relpath(path, data_dir))

    report["unsupported_g2p_chars"] = get_unsupported_char_report()["unsupported_characters"]

    for key in ("missing_transcripts", "empty_transcripts", "unknown_speaker"):
        if report[key]:
            if key == "unknown_speaker" and allow_unknown_speakers:
                warnings.append(f"{len(report[key])} clip(s) not in a speaker folder")
            else:
                blocking.append(f"{len(report[key])} {key.replace('_', ' ')}")

    if report["videos"] == 0:
        blocking.append("no video files found")
    if report["pairs"] == 0:
        blocking.append("no valid video/transcript pairs")
    if len(report["speakers"]) < 2 and not allow_unknown_speakers:
        blocking.append("fewer than 2 speaker folders (speaker-disjoint split impossible)")

    if report["pairs"]:
        report["invalid_pct"] = len(report["invalid_ctc"]) / report["pairs"] * 100.0
    if report["invalid_pct"] > max_invalid_sample_pct:
        blocking.append(
            f"invalid CTC samples {report['invalid_pct']:.1f}% > {max_invalid_sample_pct}%"
        )

    if report["non_devanagari"]:
        warnings.append(f"{len(report['non_devanagari'])} transcript(s) contain non-Devanagari characters")
    if report["unsupported_g2p_chars"]:
        warnings.append(f"unsupported G2P characters: {report['unsupported_g2p_chars']}")

    for stats in report["speakers"].values():
        stats["hours"] = round(stats["frames"] / 25.0 / 3600.0, 4)

    return report


def check_adapter(adapter_path: Optional[str], expected_base: str = "sarvamai/sarvam-1") -> dict:
    if not adapter_path:
        return {"provided": False, "compatible": False, "blocking": [], "warnings": []}

    report = {
        "provided": True,
        "path": os.path.abspath(adapter_path),
        "exists": os.path.isdir(adapter_path),
        "files": {},
        "base_model": None,
        "compatible": False,
        "blocking": [],
        "warnings": [],
    }
    if not report["exists"]:
        report["blocking"].append("adapter directory does not exist")
        return report

    config_path = os.path.join(adapter_path, "adapter_config.json")
    weights_candidates = [
        os.path.join(adapter_path, "adapter_model.safetensors"),
        os.path.join(adapter_path, "adapter_model.bin"),
    ]
    required = ["adapter_config.json"]
    for name in required:
        report["files"][name] = os.path.exists(os.path.join(adapter_path, name))
    weight_file = next((p for p in weights_candidates if os.path.exists(p)), None)
    report["files"]["adapter_model"] = bool(weight_file)
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        report["files"][name] = os.path.exists(os.path.join(adapter_path, name))

    if not report["files"]["adapter_config.json"]:
        report["blocking"].append("missing adapter_config.json")
        return report
    if not weight_file:
        report["blocking"].append("missing adapter_model.safetensors/.bin")
        return report

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        report["base_model"] = config.get("base_model_name_or_path")
        if report["base_model"] and report["base_model"] != expected_base:
            report["blocking"].append(
                f"adapter base model '{report['base_model']}' != expected '{expected_base}'"
            )
    except Exception as exc:
        report["blocking"].append(f"could not read adapter_config.json: {exc}")
        return report

    if not report["files"]["tokenizer_config.json"]:
        report["warnings"].append("adapter directory has no tokenizer files; base tokenizer will be used")

    report["compatible"] = not report["blocking"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Punarvaani dataset/adapter preflight")
    parser.add_argument("--data_dir", type=str, required=False, help="Dataset root")
    parser.add_argument("--sarvam_adapter_path", type=str, default=None)
    parser.add_argument("--sarvam_base_model", type=str, default="sarvamai/sarvam-1")
    parser.add_argument("--max_frames", type=int, default=96)
    parser.add_argument("--tubelet_size", type=int, default=2)
    parser.add_argument("--temporal_upsample_factor", type=int, default=4)
    parser.add_argument("--max_invalid_sample_pct", type=float, default=10.0)
    parser.add_argument("--allow_unknown_speakers", action="store_true")
    parser.add_argument("--output", type=str, default="preflight_report.json")
    args = parser.parse_args(argv)

    report = {"dataset": None, "adapter": None, "blocking": []}
    if args.data_dir:
        report["dataset"] = scan_dataset(
            args.data_dir,
            max_frames=args.max_frames,
            tubelet_size=args.tubelet_size,
            temporal_upsample_factor=args.temporal_upsample_factor,
            max_invalid_sample_pct=args.max_invalid_sample_pct,
            allow_unknown_speakers=args.allow_unknown_speakers,
        )
        report["blocking"].extend(report["dataset"]["blocking"])

    if args.sarvam_adapter_path:
        report["adapter"] = check_adapter(args.sarvam_adapter_path, args.sarvam_base_model)
        report["blocking"].extend(report["adapter"]["blocking"])

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"Preflight report written to {os.path.abspath(args.output)}")
    if report["dataset"]:
        ds = report["dataset"]
        print(
            f"  dataset: {ds['pairs']}/{ds['videos']} pairs, "
            f"{len(ds['speakers'])} speakers, invalid CTC {ds['invalid_pct']:.1f}%"
        )
    if report["adapter"]:
        ad = report["adapter"]
        print(f"  adapter: compatible={ad['compatible']} path={ad.get('path')}")

    if report["blocking"]:
        print("\nBLOCKING ISSUES:")
        for issue in report["blocking"]:
            print(f"  - {issue}")
        return 1
    print("Preflight OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

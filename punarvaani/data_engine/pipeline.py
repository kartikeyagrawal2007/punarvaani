"""Hindi audio-visual mining pipeline: public videos -> filtered, cropped, transcribed clips.

Per video::

    download (yt-dlp) ─► 25 fps frames + 16 kHz audio (ffmpeg)
      ─► ASR with word timestamps ─► clips at pauses (segment.py)
      ─► text filters: ASR confidence, Devanagari only, duration
      ─► one shot? (tracking.py) ─► face landmarks ─► stabilised lip crop
      ─► A/V sync check (sync.py) ─► save crop .npy + .wav + manifest row

Almost all the work in a data engine is *filtering*, so every rejected clip is
counted by reason in a :class:`FilterReport`. If 80% of clips die at the sync
check, the source is probably dubbed - drop that channel rather than tune
thresholds.

Splits are made **by speaker/channel**, never by clip: clips of the same anchor
in both train and test would measure memorisation of that anchor's lips, not
lip reading.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from punarvaani.data_engine.segment import Clip, Word, segment_words
from punarvaani.data_engine.sync import SyncResult
from punarvaani.data_engine.tracking import shot_boundaries
from punarvaani.preprocessing.audio import SAMPLE_RATE, SAMPLES_PER_VIDEO_FRAME, save_audio
from punarvaani.preprocessing.lip_crop import LipCropper
from punarvaani.text.hindi import is_mostly_devanagari, normalize

Landmarker = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]  # frames -> (landmarks, valid)
Transcriber = Callable[[torch.Tensor], list[Word]]  # 16 kHz audio -> words with timestamps
SyncScorer = Callable[[np.ndarray, torch.Tensor], SyncResult]  # (crops, audio) -> sync result


@dataclass
class FilterReport:
    kept: int = 0
    dropped: Counter = field(default_factory=Counter)
    kept_seconds: float = 0.0

    def drop(self, reason: str) -> None:
        self.dropped[reason] += 1

    def merge(self, other: FilterReport) -> None:
        self.kept += other.kept
        self.kept_seconds += other.kept_seconds
        self.dropped.update(other.dropped)

    def summary(self) -> dict:
        total = self.kept + sum(self.dropped.values())
        return {"total_clips": total, "kept": self.kept, "kept_hours": round(self.kept_seconds / 3600, 3),
                "dropped": dict(self.dropped.most_common())}  # fmt: skip


@dataclass
class MiningConfig:
    fps: int = 25
    min_clip_s: float = 1.0
    max_clip_s: float = 12.0
    min_asr_confidence: float = 0.6
    min_devanagari: float = 0.9
    min_sync_confidence: float = 3.0
    max_sync_offset: int = 3
    shot_threshold: float = 0.4


class MiningPipeline:
    def __init__(self, out_dir: str | Path, landmarker: Landmarker, transcriber: Transcriber,
                 sync_scorer: SyncScorer | None = None, cfg: MiningConfig | None = None,
                 cropper: LipCropper | None = None):  # fmt: skip
        self.out_dir = Path(out_dir)
        self.landmarker, self.transcriber, self.sync_scorer = landmarker, transcriber, sync_scorer
        self.cfg = cfg or MiningConfig()
        self.cropper = cropper or LipCropper()
        for sub in ("crops", "wav"):
            (self.out_dir / sub).mkdir(parents=True, exist_ok=True)

    # --- text-level filters -------------------------------------------------
    def _text_ok(self, clip: Clip, report: FilterReport) -> bool:
        if clip.duration > self.cfg.max_clip_s:
            report.drop("too_long")
        elif clip.min_confidence < self.cfg.min_asr_confidence:
            report.drop("low_asr_confidence")
        elif not is_mostly_devanagari(clip.text, self.cfg.min_devanagari):
            report.drop("code_mixed_or_non_hindi")
        elif not normalize(clip.text):
            report.drop("empty_text")
        else:
            return True
        return False

    def process_video(self, video_id: str, frames: np.ndarray, audio: torch.Tensor, speaker: str,
                      source: str = "") -> tuple[list[dict], FilterReport]:  # fmt: skip
        """``frames``: (T, H, W, 3) RGB at 25 fps; ``audio``: 16 kHz mono aligned with the frames."""
        report = FilterReport()
        rows: list[dict] = []
        cuts = shot_boundaries(frames, threshold=self.cfg.shot_threshold)
        words = self.transcriber(audio)
        for k, clip in enumerate(segment_words(words, self.cfg.min_clip_s, self.cfg.max_clip_s)):
            if not self._text_ok(clip, report):
                continue
            f0, f1 = int(round(clip.start * self.cfg.fps)), int(round(clip.end * self.cfg.fps))
            f1 = min(f1, len(frames))
            if f1 - f0 < self.cfg.min_clip_s * self.cfg.fps:
                report.drop("too_short_after_alignment")
                continue
            if any(f0 < c < f1 for c in cuts):
                report.drop("shot_cut_inside_clip")
                continue
            lms, valid = self.landmarker(frames[f0:f1])
            try:
                crop = self.cropper(frames[f0:f1], lms, valid)
            except ValueError:
                report.drop("face_not_found")
                continue
            wav = audio[f0 * SAMPLES_PER_VIDEO_FRAME : f1 * SAMPLES_PER_VIDEO_FRAME]
            if self.sync_scorer is not None:
                sync = self.sync_scorer(crop.crops, wav)
                if not sync.ok(self.cfg.min_sync_confidence, self.cfg.max_sync_offset):
                    report.drop("out_of_sync")
                    continue
                if sync.offset:  # shift audio so lips and sound line up exactly
                    shift = sync.offset * SAMPLES_PER_VIDEO_FRAME
                    wav = torch.roll(wav, -shift)
            clip_id = f"{video_id}_{k:04d}"
            np.save(self.out_dir / "crops" / f"{clip_id}.npy", crop.crops)
            try:
                save_audio(self.out_dir / "wav" / f"{clip_id}.wav", wav)
                audio_path = f"wav/{clip_id}.wav"
            except ImportError:
                torch.save(wav, self.out_dir / "wav" / f"{clip_id}.pt")
                audio_path = f"wav/{clip_id}.pt"
            rows.append({"id": clip_id, "video": f"crops/{clip_id}.npy", "audio": audio_path,
                         "text": normalize(clip.text), "speaker": speaker, "mode": "spoken",
                         "num_frames": int(crop.crops.shape[0]), "source": source})  # fmt: skip
            report.kept += 1
            report.kept_seconds += crop.crops.shape[0] / self.cfg.fps
        return rows, report


def split_by_group(rows: Sequence[dict], key: str = "speaker", val: float = 0.05, test: float = 0.05,
                   salt: str = "punarvaani") -> dict[str, list[dict]]:  # fmt: skip
    """Deterministic speaker-disjoint split by hashing the group id (stable as data grows)."""
    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in rows:
        h = int(hashlib.sha1(f"{salt}:{r.get(key, '')}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        out["test" if h < test else "val" if h < test + val else "train"].append(r)
    return out


# --- external tools (not needed for tests) -------------------------------------


def download(url: str, out_dir: str | Path) -> Path:
    """Download one video with yt-dlp (respect each source's terms of use)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / "%(id)s.%(ext)s")
    subprocess.run(["yt-dlp", "-f", "mp4[height<=720]/best", "-o", template, url], check=True)
    vid = subprocess.run(
        ["yt-dlp", "--get-id", url], check=True, capture_output=True, text=True
    ).stdout.strip()
    return next(out_dir.glob(f"{vid}.*"))


def extract_audio(video_path: str | Path, wav_path: str | Path) -> Path:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path), "-ac", "1", "-ar", str(SAMPLE_RATE),
                    str(wav_path)], check=True)  # fmt: skip
    return Path(wav_path)


class HFTranscriber:
    """Word-timestamped ASR with a Hugging Face Whisper-family checkpoint.

    Point ``model_name`` at an AI4Bharat Hindi checkpoint (IndicWhisper) for
    much better Hindi accuracy than multilingual Whisper.
    """

    def __init__(
        self, model_name: str = "openai/whisper-large-v3", device: str | int = -1, language: str = "hi"
    ):
        from transformers import pipeline

        self.asr = pipeline("automatic-speech-recognition", model=model_name, device=device)
        self.language = language

    def __call__(self, audio: torch.Tensor) -> list[Word]:
        out = self.asr(audio.numpy(), return_timestamps="word", generate_kwargs={"language": self.language},
                       chunk_length_s=30)  # fmt: skip
        words = []
        for ch in out.get("chunks", []):
            start, end = ch["timestamp"]
            if start is None or end is None:
                continue
            words.append(Word(ch["text"].strip(), float(start), float(end)))
        return words


def write_splits(rows: Sequence[dict], out_dir: str | Path, report: FilterReport | None = None) -> None:
    from punarvaani.datasets.av_dataset import write_manifest

    out_dir = Path(out_dir)
    for name, part in split_by_group(rows).items():
        write_manifest(part, out_dir / f"{name}.jsonl")
    if report is not None:
        (out_dir / "filter_report.json").write_text(json.dumps(report.summary(), indent=1))

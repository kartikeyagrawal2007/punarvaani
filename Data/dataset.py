"""
Hindi Visual Speech Recognition dataset.

On-disk contract:
    data_dir/<speaker>/<clip>.mp4
    data_dir/<speaker>/<clip>.txt     (UTF-8 Devanagari)

Behaviour:
- speaker-disjoint train/val splitting (first path component = speaker),
- all frames are read (optionally strided and capped by ``max_frames``),
- optional mouth / face / no cropping,
- per-sample CTC time-step accounting (``ceil(frames/tubelet) * upsample``),
- optional pre-check that no more than ``max_invalid_sample_pct`` of samples
  violate the CTC ``T >= U + R`` constraint.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from decord import VideoReader, cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

from Data.hindi_g2p import devanagari_to_phonemes

try:
    from face_cropper import FaceCropper
    HAS_FACE_CROPPER = True
except ImportError:
    HAS_FACE_CROPPER = False


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


@dataclass
class VSRItem:
    """One variable-length video/phoneme sample."""

    video: torch.Tensor          # (T, C, H, W), float32 in [0, 1]
    phoneme_ids: torch.Tensor    # (U,), long
    text: str
    speaker_id: str
    num_frames: int


@dataclass
class VSRBatch:
    """Padded batch produced by :func:`custom_collate`."""

    videos: Optional[torch.Tensor]          # (B, T_max, C, H, W)
    labels: List[torch.Tensor]
    texts: List[str]
    video_lengths: Optional[torch.Tensor]   # (B,)
    input_lengths: Optional[torch.Tensor]   # (B,)
    valid_mask: Optional[torch.Tensor]      # (B,) bool
    dropped: int = 0

    def __len__(self) -> int:
        return 0 if self.videos is None else int(self.videos.size(0))


def compute_ctc_min_length(label_tensor: torch.Tensor) -> int:
    """Minimum CTC time steps: U + R (target length plus adjacent repeats)."""
    if len(label_tensor) == 0:
        return 0
    u = len(label_tensor)
    r = int((label_tensor[:-1] == label_tensor[1:]).sum().item()) if u > 1 else 0
    return u + r


def speaker_id_for_path(video_path: str, data_dir: str) -> Optional[str]:
    """
    Speaker = first path component under ``data_dir``.

    Returns None when the file sits directly in ``data_dir``.
    """
    rel = os.path.relpath(video_path, data_dir)
    parts = os.path.normpath(rel).split(os.sep)
    if len(parts) <= 1:
        return None
    return parts[0]


def _probe_frame_count(video_path: str) -> int:
    """Cheap frame-count probe (0 when the container cannot report it)."""
    if HAS_DECORD:
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            count = len(vr)
            if count > 0:
                return int(count)
        except Exception:
            pass
    cap = cv2.VideoCapture(video_path)
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return max(0, count)


def _read_all_frames_opencv(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


class HindiVideoDataset(Dataset):
    """Speaker-disjoint, variable-length Hindi VSR dataset."""

    def __init__(
        self,
        data_dir: str,
        phoneme_vocab: Dict[str, int],
        split: str = "train",
        val_split: float = 0.1,
        sample_size: float = 1.0,
        is_dedicated_split: bool = False,
        num_frames: int = 0,
        max_frames: int = 96,
        frame_stride: int = 1,
        frame_size: Tuple[int, int] = (224, 224),
        crop_mode: str = "mouth",
        preprocessed: bool = False,
        seed: int = 42,
        allow_unknown_speakers: bool = False,
        max_invalid_sample_pct: float = 10.0,
        strict_ctc: bool = False,
        validate_ctc: bool = True,
        tubelet_size: int = 2,
        temporal_upsample_factor: int = 4,
    ):
        self.data_dir = data_dir
        self.phoneme_vocab = phoneme_vocab
        self.split = split
        self.sample_size = float(sample_size)
        self.num_frames = int(num_frames or 0)
        self.max_frames = int(max_frames)
        self.frame_stride = max(1, int(frame_stride))
        self.frame_size = frame_size
        self.crop_mode = "none" if preprocessed else crop_mode
        self.allow_unknown_speakers = bool(allow_unknown_speakers)
        self.max_invalid_sample_pct = float(max_invalid_sample_pct)
        self.strict_ctc = bool(strict_ctc)
        self.validate_ctc = bool(validate_ctc)
        self.tubelet_size = int(tubelet_size)
        self.temporal_upsample_factor = int(temporal_upsample_factor)

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")

        # ---- discover paired samples -------------------------------------
        discovered: List[dict] = []
        for root, _, files in os.walk(data_dir):
            for name in files:
                if not name.lower().endswith(VIDEO_EXTENSIONS):
                    continue
                base, _ = os.path.splitext(name)
                video_path = os.path.join(root, name)
                txt_path = os.path.join(root, base + ".txt")
                if not os.path.exists(txt_path):
                    continue
                discovered.append(
                    {
                        "video": video_path,
                        "txt": txt_path,
                        "speaker": speaker_id_for_path(video_path, data_dir),
                    }
                )

        if not discovered:
            raise ValueError(
                f"No valid video/transcript pairs found in '{data_dir}'. "
                "Expected <speaker>/<clip>.mp4 with a matching .txt file."
            )

        discovered.sort(key=lambda item: item["video"])
        unknown = [item for item in discovered if item["speaker"] is None]
        if unknown and not self.allow_unknown_speakers:
            example = os.path.basename(unknown[0]["video"])
            raise ValueError(
                f"{len(unknown)} clip(s) are not inside a speaker subfolder (e.g. '{example}'). "
                "Arrange the dataset as '<data_dir>/<speaker>/<clip>.mp4' or pass "
                "--allow_unknown_speakers to fall back to a random clip split."
            )

        selected = self._select_split(discovered, split, val_split, is_dedicated_split, seed)
        selected = self._subsample(selected, seed)

        # ---- annotation + CTC pre-check ----------------------------------
        self.samples: List[dict] = []
        invalid = 0
        for item in selected:
            text = self._read_transcript(item["txt"])
            phoneme_tokens = devanagari_to_phonemes(text)
            phoneme_ids = [
                phoneme_vocab[token] for token in phoneme_tokens if token in phoneme_vocab
            ]
            if not phoneme_ids:
                phoneme_ids = [phoneme_vocab.get("A", 1)]
            label = torch.tensor(phoneme_ids, dtype=torch.long)
            ctc_min = compute_ctc_min_length(label)

            total_frames = _probe_frame_count(item["video"]) if self.validate_ctc else 0
            if self.num_frames > 0:
                used_frames = self.num_frames
            elif total_frames > 0:
                used_frames = min(total_frames, self.max_frames)
            else:
                used_frames = self.max_frames
            t_ctc = math.ceil(used_frames / self.tubelet_size) * self.temporal_upsample_factor

            item = dict(item)
            item.update(
                {
                    "text": text,
                    "label": label,
                    "ctc_min": ctc_min,
                    "estimated_frames": used_frames,
                    "estimated_input_length": t_ctc,
                }
            )
            self.samples.append(item)
            if ctc_min > t_ctc:
                invalid += 1

        self.invalid_count = invalid
        self.invalid_pct = (invalid / len(self.samples) * 100.0) if self.samples else 0.0
        self._report_ctc_health()

        # ---- face cropper -------------------------------------------------
        self.face_cropper = None
        if self.crop_mode != "none":
            if not HAS_FACE_CROPPER:
                raise RuntimeError(
                    f"crop_mode='{self.crop_mode}' requires MediaPipe, but it is not installed. "
                    "Install it with `pip install mediapipe`, or pass --preprocessed "
                    "(if the clips are already mouth-cropped) / --crop_mode none."
                )
            try:
                self.face_cropper = FaceCropper(
                    min_face_detector_confidence=0.5,
                    face_detector_model_selection="SHORT_RANGE",
                    landmark_detector_static_image_mode="STATIC_MODE",
                    min_landmark_detector_confidence=0.5,
                )
            except Exception as exc:  # pragma: no cover - env dependent
                raise RuntimeError(
                    f"Could not initialise FaceCropper for crop_mode='{self.crop_mode}': {exc}"
                ) from exc

        print(
            f"[HindiVideoDataset] Loaded {len(self.samples)} samples for '{split}' "
            f"from '{data_dir}' (crop_mode={self.crop_mode}, "
            f"frames={'uniform ' + str(self.num_frames) if self.num_frames else 'all up to ' + str(self.max_frames)})."
        )

    # ------------------------------------------------------------------ helpers
    def _report_ctc_health(self) -> None:
        if self.invalid_count == 0:
            return
        message = (
            f"[HindiVideoDataset] {self.invalid_count}/{len(self.samples)} "
            f"({self.invalid_pct:.1f}%) samples violate CTC T>=U+R and will be dropped."
        )
        if self.invalid_pct > self.max_invalid_sample_pct:
            if self.strict_ctc:
                raise RuntimeError(
                    message
                    + f" Exceeds --max_invalid_sample_pct={self.max_invalid_sample_pct}. "
                    "Shorten clips, raise --max_frames, or lower --temporal_upsample_factor."
                )
            print(message + f" (above --max_invalid_sample_pct={self.max_invalid_sample_pct})")
        else:
            print(message)

    @staticmethod
    def _read_transcript(txt_path: str) -> str:
        with open(txt_path, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
        return " ".join(lines)

    def _select_split(
        self,
        samples: List[dict],
        split: str,
        val_split: float,
        is_dedicated_split: bool,
        seed: int,
    ) -> List[dict]:
        self.split_speakers = []
        if is_dedicated_split or val_split <= 0.0 or len(samples) < 2:
            return list(samples)

        speakers = sorted({item["speaker"] for item in samples if item["speaker"] is not None})
        if speakers and len(speakers) < 2:
            raise ValueError(
                "Speaker-disjoint validation needs at least two speaker folders. "
                "Add more speakers or pass a dedicated --val_dir."
            )
        if speakers and len(speakers) >= 2:
            rng = np.random.RandomState(seed)
            order = rng.permutation(len(speakers))
            n_val = max(1, int(round(len(speakers) * val_split)))
            n_val = min(n_val, len(speakers) - 1)  # never empty the train side
            val_speakers = {speakers[i] for i in order[:n_val]}
            train_speakers = set(speakers) - val_speakers

            leakage = train_speakers.intersection(val_speakers)
            if leakage:  # pragma: no cover - defensive
                raise ValueError(f"Speaker leakage detected: {sorted(leakage)}")

            chosen = val_speakers if split == "val" else train_speakers
            self.split_speakers = sorted(chosen)
            return [item for item in samples if item["speaker"] in chosen]

        # No speaker information: fall back to a deterministic clip-level split.
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(samples))
        val_count = max(1, int(len(samples) * val_split))
        val_count = min(val_count, len(samples) - 1)
        chosen_idx = indices[:val_count] if split == "val" else indices[val_count:]
        self.split_speakers = []
        return [samples[i] for i in chosen_idx]

    def _subsample(self, samples: List[dict], seed: int) -> List[dict]:
        if not (0.0 < self.sample_size < 1.0):
            return samples
        count = max(1, int(len(samples) * self.sample_size))
        rng = np.random.RandomState(seed + 1)
        indices = rng.permutation(len(samples))[:count]
        return [samples[i] for i in indices]

    # ------------------------------------------------------------------ video
    def _frame_indices(self, total_frames: int) -> np.ndarray:
        if total_frames <= 0:
            return np.zeros((0,), dtype=np.int64)
        if self.num_frames > 0:
            count = min(self.num_frames, total_frames)
            if count >= total_frames:
                return np.arange(total_frames, dtype=np.int64)
            return np.linspace(0, total_frames - 1, count, dtype=np.int64)
        indices = np.arange(0, total_frames, self.frame_stride, dtype=np.int64)
        return indices[: self.max_frames]

    def _read_frames(self, video_path: str) -> Optional[np.ndarray]:
        if HAS_DECORD:
            try:
                reader = VideoReader(video_path, ctx=cpu(0), num_threads=2)
                indices = self._frame_indices(len(reader))
                if len(indices) == 0:
                    return None
                return reader.get_batch(indices).asnumpy()
            except Exception:
                pass

        frames = _read_all_frames_opencv(video_path)
        if not frames:
            return None
        indices = self._frame_indices(len(frames))
        if len(indices) == 0:
            return None
        return np.stack([frames[i] for i in indices], axis=0)

    def _crop_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.crop_mode == "mouth" and self.face_cropper is not None:
            crop = self.face_cropper.crop_mouth(frame, frame_size=self.frame_size)
        elif self.crop_mode == "face" and self.face_cropper is not None:
            faces = self.face_cropper.get_faces(frame, remove_background=False, correct_roll=True)
            crop = faces[0] if faces else None
        else:
            crop = frame

        if crop is None or crop.size == 0:
            crop = frame
        return cv2.resize(crop, self.frame_size)

    def _load_video_frames(self, video_path: str) -> Optional[np.ndarray]:
        frames = self._read_frames(video_path)
        if frames is None or len(frames) == 0:
            return None

        if self.face_cropper is not None and hasattr(self.face_cropper, "reset"):
            self.face_cropper.reset()

        processed = [self._crop_frame(frame) for frame in frames]
        if len(processed) < self.tubelet_size:
            processed.extend([processed[-1]] * (self.tubelet_size - len(processed)))
        return np.stack(processed, axis=0)

    # ------------------------------------------------------------------ Dataset
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[VSRItem]:
        sample = self.samples[idx]
        frames_np = self._load_video_frames(sample["video"])
        if frames_np is None:
            return None

        video = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0
        return VSRItem(
            video=video,
            phoneme_ids=sample["label"].clone(),
            text=sample["text"],
            speaker_id=sample["speaker"] or "",
            num_frames=int(video.size(0)),
        )


def custom_collate(
    batch: Sequence,
    phoneme_vocab: Optional[Dict[str, int]] = None,
    t_ctc: Optional[int] = None,
    tubelet_size: int = 2,
    temporal_upsample_factor: int = 4,
) -> VSRBatch:
    """
    Pad variable-length videos and compute per-sample CTC lengths.

    ``t_ctc`` forces a fixed number of CTC steps (legacy callers). Otherwise it is
    derived per sample from the number of frames.
    """
    items = [item for item in batch if item is not None]
    prepared: List[dict] = []
    for item in items:
        if isinstance(item, VSRItem):
            video = item.video
            label = item.phoneme_ids
            text = item.text
            num_frames = item.num_frames
        else:
            video, label, text = item[0], item[1], item[2]
            num_frames = int(video.size(0))

        if t_ctc is not None:
            input_length = int(t_ctc)
        else:
            tubelets = max(1, int(math.ceil(num_frames / tubelet_size)))
            input_length = tubelets * temporal_upsample_factor

        if input_length >= compute_ctc_min_length(label):
            prepared.append(
                {
                    "video": video,
                    "label": label,
                    "text": text,
                    "input_length": input_length,
                    "num_frames": num_frames,
                }
            )

    dropped = len(items) - len(prepared)
    if not prepared:
        return VSRBatch(None, [], [], None, None, None, dropped=dropped)

    max_frames = max(item["num_frames"] for item in prepared)
    channels, height, width = prepared[0]["video"].shape[1:]
    videos = torch.zeros(len(prepared), max_frames, channels, height, width)
    video_lengths = torch.zeros(len(prepared), dtype=torch.long)
    input_lengths = torch.zeros(len(prepared), dtype=torch.long)
    max_input_length = int(math.ceil(max_frames / tubelet_size)) * temporal_upsample_factor

    for i, item in enumerate(prepared):
        frames = item["num_frames"]
        videos[i, :frames] = item["video"]
        video_lengths[i] = frames
        input_lengths[i] = min(item["input_length"], max_input_length)

    return VSRBatch(
        videos=videos,
        labels=[item["label"].clone().detach() for item in prepared],
        texts=[item["text"] for item in prepared],
        video_lengths=video_lengths,
        input_lengths=input_lengths,
        valid_mask=torch.ones(len(prepared), dtype=torch.bool),
        dropped=dropped,
    )


# ---------------------------------------------------------------------------
# Speaker-leakage helpers (kept for the existing test / external callers)
# ---------------------------------------------------------------------------
def check_speaker_leakage(train_samples, val_samples, speaker_extractor=None):
    """Return speaker IDs present in both train and validation sets."""
    def default_extractor(item):
        path = item[0] if isinstance(item, (tuple, list)) else str(item)
        parent = os.path.basename(os.path.dirname(path))
        base = os.path.splitext(os.path.basename(path))[0]
        m_parent = re.match(r"^(?:spk|speaker)[-_]?([a-zA-Z0-9]+)", parent, re.IGNORECASE)
        if m_parent:
            return m_parent.group(0).lower()
        m_base = re.match(r"^((?:spk|speaker)[-_]?[a-zA-Z0-9]+)", base, re.IGNORECASE)
        if m_base:
            return m_base.group(1).lower()
        return parent.lower() if parent else base.lower()

    extractor = speaker_extractor or default_extractor
    train_speakers = {extractor(sample) for sample in train_samples}
    val_speakers = {extractor(sample) for sample in val_samples}
    return train_speakers.intersection(val_speakers)


def verify_speaker_disjoint_split(train_samples, val_samples, speaker_extractor=None):
    """Raise ``ValueError`` when any speaker appears in both splits."""
    leakage = check_speaker_leakage(train_samples, val_samples, speaker_extractor)
    if leakage:
        raise ValueError(
            f"Speaker leakage detected! Speakers present in both train and val: {sorted(leakage)}"
        )
    return True


# Backward compatibility alias
VideoDataset = HindiVideoDataset

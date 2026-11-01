import os
import re
import cv2
import torch
import numpy as np
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


class HindiVideoDataset(Dataset):
    """
    Dataset for Hindi Visual Speech Recognition.
    Loads paired video files (dataXXX.mp4) and transcripts (dataXXX.txt) from a directory.
    """
    def __init__(
        self,
        data_dir: str,
        phoneme_vocab: dict,
        split: str = "train",
        val_split: float = 0.1,
        sample_size: float = 1.0,
        is_dedicated_split: bool = False,
        num_frames: int = 16,
        frame_size: tuple = (224, 224),
        crop_face: bool = True,
        seed: int = 42
    ):
        self.data_dir = data_dir
        self.phoneme_vocab = phoneme_vocab
        self.split = split
        self.sample_size = sample_size
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.crop_face = crop_face

        self.face_cropper = None
        if self.crop_face and HAS_FACE_CROPPER:
            try:
                self.face_cropper = FaceCropper(
                    min_face_detector_confidence=0.5,
                    face_detector_model_selection="SHORT_RANGE",
                    landmark_detector_static_image_mode="STATIC_MODE",
                    min_landmark_detector_confidence=0.5
                )
            except Exception as e:
                print(f"[Warning] Could not initialize FaceCropper: {e}. Falling back to center crop.")
        elif self.crop_face and not HAS_FACE_CROPPER:
            print("[Warning] FaceCropper (MediaPipe) not installed/available. Using center crop fallback.")

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")

        # Discover all paired videos and transcripts
        all_samples = []
        valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    base_name = os.path.splitext(file)[0]
                    txt_name = f"{base_name}.txt"
                    txt_path = os.path.join(root, txt_name)
                    video_path = os.path.join(root, file)

                    if os.path.exists(txt_path):
                        all_samples.append((video_path, txt_path))

        if len(all_samples) == 0:
            raise ValueError(f"No valid video-transcript pairs found in '{data_dir}'. Expected matching .mp4 and .txt files.")

        # Deterministic sorting
        all_samples.sort(key=lambda x: x[0])
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_samples))

        # Train/val split handling (respects dedicated val directory)
        if not is_dedicated_split and val_split > 0.0 and len(all_samples) > 1:
            val_count = max(1, int(len(all_samples) * val_split))
            if val_count >= len(all_samples):
                val_count = len(all_samples) - 1
            if split == "val":
                selected_indices = indices[:val_count]
            else:
                selected_indices = indices[val_count:]
        else:
            selected_indices = indices

        # Subsampling via sample_size
        if 0.0 < self.sample_size < 1.0:
            sub_count = max(1, int(len(selected_indices) * self.sample_size))
            selected_indices = selected_indices[:sub_count]

        self.samples = [all_samples[i] for i in selected_indices]
        print(f"[HindiVideoDataset] Loaded {len(self.samples)} samples for '{split}' split from '{data_dir}' (sample_size={self.sample_size}).")

    def __len__(self):
        return len(self.samples)

    def _read_transcript(self, txt_path: str) -> str:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            return " ".join(lines)

    def _load_video_frames(self, video_path: str) -> np.ndarray:
        """Loads video and samples num_frames evenly, applying stabilized face crop."""
        vr = None
        frame_count = 0

        if HAS_DECORD:
            try:
                vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
                frame_count = len(vr)
            except Exception:
                vr = None

        if vr is None:
            # Fallback to OpenCV VideoCapture
            cap = cv2.VideoCapture(video_path)
            frames_raw = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames_raw.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            frame_count = len(frames_raw)
            vr = frames_raw


        if frame_count < 1:
            return None

        # Evenly sample frame indices
        if frame_count >= self.num_frames:
            sample_indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)
        else:
            sample_indices = np.pad(
                np.arange(frame_count),
                (0, self.num_frames - frame_count),
                mode='edge'
            )

        sampled_frames = []
        for idx in sample_indices:
            if isinstance(vr, list):
                frame = vr[idx]
            else:
                frame = vr[idx].asnumpy()
            sampled_frames.append(frame)

        # Apply face cropping with bounding box stabilization
        processed_frames = []
        last_valid_crop = None

        for frame in sampled_frames:
            cropped = None
            if self.face_cropper is not None:
                try:
                    faces = self.face_cropper.get_faces(frame, remove_background=False, correct_roll=True)
                    if faces and len(faces) > 0 and faces[0].size > 0:
                        cropped = faces[0]
                        last_valid_crop = cropped
                except Exception:
                    pass

            if cropped is None:
                # Reuse last valid face crop if available to prevent jitter scale jump
                if last_valid_crop is not None:
                    cropped = last_valid_crop
                else:
                    cropped = frame

            cropped = cv2.resize(cropped, self.frame_size)
            processed_frames.append(cropped)

        video_np = np.stack(processed_frames, axis=0)  # (T, H, W, C)
        return video_np

    def __getitem__(self, idx: int):
        video_path, txt_path = self.samples[idx]

        # 1. Load and process video frames
        video_np = self._load_video_frames(video_path)
        if video_np is None:
            # Return dummy on failure
            video_np = np.zeros((self.num_frames, self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)

        # Convert to Tensor (T, C, H, W) in range [0, 1]
        video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0

        # 2. Load transcript and convert to Hindi phonemes
        text = self._read_transcript(txt_path)
        phoneme_tokens = devanagari_to_phonemes(text)

        # Map to vocabulary indices
        phoneme_indices = [
            self.phoneme_vocab[token]
            for token in phoneme_tokens
            if token in self.phoneme_vocab
        ]

        # Ensure target sequence is non-empty
        if len(phoneme_indices) == 0:
            phoneme_indices = [self.phoneme_vocab.get('A', 1)]

        phoneme_tensor = torch.tensor(phoneme_indices, dtype=torch.long)

        return video_tensor, phoneme_tensor, text


def check_speaker_leakage(train_samples, val_samples, speaker_extractor=None):
    """
    Checks for speaker identity leakage between train and validation sample sets.
    Returns the set of speaker IDs that appear in both splits.
    """
    def default_extractor(item):
        path = item[0] if isinstance(item, (tuple, list)) else str(item)
        parent = os.path.basename(os.path.dirname(path))
        base = os.path.splitext(os.path.basename(path))[0]
        m_parent = re.match(r'^(?:spk|speaker)[-_]?([a-zA-Z0-9]+)', parent, re.IGNORECASE)
        if m_parent:
            return m_parent.group(0).lower()
        m_base = re.match(r'^((?:spk|speaker)[-_]?[a-zA-Z0-9]+)', base, re.IGNORECASE)
        if m_base:
            return m_base.group(1).lower()
        return parent.lower() if parent else base.lower()

    extractor = speaker_extractor or default_extractor
    train_speakers = {extractor(s) for s in train_samples}
    val_speakers = {extractor(s) for s in val_samples}
    return train_speakers.intersection(val_speakers)


def verify_speaker_disjoint_split(train_samples, val_samples, speaker_extractor=None):
    """
    Raises ValueError if any speaker is present in both train and validation splits.
    """
    leakage = check_speaker_leakage(train_samples, val_samples, speaker_extractor)
    if leakage:
        raise ValueError(f"Speaker leakage detected! Speakers present in both train and val: {sorted(list(leakage))}")
    return True


def compute_ctc_min_length(label_tensor: torch.Tensor) -> int:
    """Computes minimum CTC length U + R for a target label sequence."""
    if len(label_tensor) == 0:
        return 0
    u = len(label_tensor)
    r = int((label_tensor[:-1] == label_tensor[1:]).sum().item()) if u > 1 else 0
    return u + r


def custom_collate(batch, phoneme_vocab, t_ctc: int = 32):
    """
    Custom collator for HindiVideoDataset with per-sample CTC validity check (T >= U + R).
    Stacks videos and packs phoneme labels. Excludes invalid samples.
    If all samples are invalid or empty, returns (None, [], []).
    """
    valid_batch = [item for item in batch if item is not None and item[0] is not None]
    if len(valid_batch) == 0:
        return None, [], []

    ctc_valid_batch = []
    for item in valid_batch:
        video, label, text = item
        min_required = compute_ctc_min_length(label)
        if t_ctc >= min_required:
            ctc_valid_batch.append(item)
        else:
            print(f"[CTC Filter Warning] Sample with target len U={len(label)}, min_required={min_required} dropped (T_ctc={t_ctc})")

    if len(ctc_valid_batch) == 0:
        return None, [], []

    videos, labels, texts = zip(*ctc_valid_batch)
    videos_tensor = torch.stack(videos)  # (B, T, C, H, W)
    label_tensors = [l.clone().detach() for l in labels]

    return videos_tensor, label_tensors, list(texts)


# Backward compatibility alias
VideoDataset = HindiVideoDataset
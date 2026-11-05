"""
Punarvaani Inference Interface
Backend-independent, production-ready inference API for Hindi Visual Speech Recognition.
Returns structured dictionaries with clear status indicators, error handling, and mock indicators.
"""

import os
from typing import Any, Dict, Optional, Tuple

import torch

from Models.ML_VALLR import ML_VALLR
from Models.Sarvam import HindiPhonemeTranslator
from config import get_hindi_vocab


def load_inference_video(
    video_path: str,
    num_frames: int = 0,
    max_frames: int = 96,
    frame_stride: int = 1,
    frame_size: Tuple[int, int] = (224, 224),
    crop_mode: str = "mouth",
) -> Tuple[Optional[torch.Tensor], int]:
    """
    Extract, crop and normalise frames from a video.

    Returns ``(video_tensor, length)`` where the tensor is ``(1, T, C, H, W)``
    in ``[0, 1]`` and ``length`` is the number of frames. ``(None, 0)`` on failure.
    """
    from Data.dataset import HindiVideoDataset

    dataset_helper = HindiVideoDataset.__new__(HindiVideoDataset)
    dataset_helper.num_frames = int(num_frames or 0)
    dataset_helper.max_frames = int(max_frames)
    dataset_helper.frame_stride = max(1, int(frame_stride))
    dataset_helper.frame_size = frame_size
    dataset_helper.crop_mode = crop_mode
    dataset_helper.tubelet_size = 2
    dataset_helper.face_cropper = None

    if crop_mode != "none":
        try:
            from face_cropper import FaceCropper

            dataset_helper.face_cropper = FaceCropper(
                min_face_detector_confidence=0.5,
                face_detector_model_selection="SHORT_RANGE",
                landmark_detector_static_image_mode="STATIC_MODE",
                min_landmark_detector_confidence=0.5,
            )
        except Exception as exc:
            raise RuntimeError(
                f"crop_mode='{crop_mode}' requires MediaPipe, but it is unavailable: {exc}. "
                "Install it with `pip install mediapipe` or pass crop_mode='none' "
                "for an already-cropped video."
            ) from exc

    video_np = dataset_helper._load_video_frames(video_path)
    if video_np is None:
        return None, 0

    # (T, H, W, C) -> (T, C, H, W) -> (1, T, C, H, W)
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0
    return video_tensor.unsqueeze(0), int(video_np.shape[0])


def transcribe_video(
    video_path: str,
    stage1_model_path: Optional[str] = None,
    sarvam_adapter_path: Optional[str] = None,
    device: Optional[str] = None,
    crop_face: bool = True,
    temporal_upsample_factor: int = 4,
    max_frames: int = 96,
    num_frames: int = 0,
    frame_stride: int = 1,
    crop_mode: Optional[str] = None,
    vit_embed_dim: int = 768,
    vit_depth: int = 8,
    vit_heads: int = 12,
    spatial_pool_tokens: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """
    Unified, framework-independent inference entry point for Hindi Visual Speech Recognition.

    Args:
        video_path: Path to the input video file (e.g. .mp4, .avi, .mkv).
        stage1_model_path: Path to Stage 1 ML_VALLR trained weights (.pth).
        sarvam_adapter_path: Path to a PEFT LoRA adapter directory for Sarvam-1.
        device: 'cuda', 'cpu', or None (auto-detect).
        crop_face: Whether to apply MediaPipe mouth tracking (ignored when crop_mode is set).
        temporal_upsample_factor: Upsampling factor in the temporal adapter.
        max_frames: Cap on frames read from the clip.
        num_frames: Uniformly sample this many frames; 0 means all (up to max_frames).
        frame_stride: Read every Nth frame.
        crop_mode: 'mouth' | 'face' | 'none'; defaults from crop_face.
        vit_embed_dim / vit_depth / vit_heads: Stage-1 architecture (must match weights).
        spatial_pool_tokens: ViT spatial pooling grid used during training, or None.

    Returns:
        Dict with keys: success, phonemes, transcription, stage1_status,
        stage2_status, error, num_frames.
    """
    if crop_mode is None:
        crop_mode = "mouth" if crop_face else "none"

    result: Dict[str, Any] = {
        "success": False,
        "phonemes": "",
        "transcription": None,
        "stage1_status": "failed",
        "stage2_status": "adapter_missing" if not sarvam_adapter_path else "failed",
        "error": None,
        "num_frames": 0,
    }

    if not video_path or not os.path.exists(video_path):
        result["error"] = f"Video file not found at: '{video_path}'"
        return result

    dev = torch.device(
        device
        if device and torch.cuda.is_available() and "cuda" in device
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    vocab = get_hindi_vocab()

    # Step 1: Preprocessing & frame extraction
    try:
        video_tensor, num_frames_read = load_inference_video(
            video_path,
            num_frames=num_frames,
            max_frames=max_frames,
            frame_stride=frame_stride,
            crop_mode=crop_mode,
        )
        if video_tensor is None:
            result["error"] = "Failed to decode video frames."
            return result
        video_tensor = video_tensor.to(dev)
        video_lengths = torch.tensor([num_frames_read], dtype=torch.long, device=dev)
        result["num_frames"] = num_frames_read
    except Exception as exc:
        result["error"] = f"Preprocessing error: {exc}"
        return result

    # Step 2: Stage 1 Visual-to-Phoneme Model
    try:
        stage1_model = ML_VALLR(
            adapter_dim=256,
            num_classes=len(vocab),
            temporal_upsample_factor=temporal_upsample_factor,
            feature_size=vit_embed_dim,
            vit_depth=vit_depth,
            vit_heads=vit_heads,
            spatial_pool_tokens=spatial_pool_tokens,
        )
        if stage1_model_path and os.path.exists(stage1_model_path):
            try:
                state_dict = torch.load(stage1_model_path, map_location=dev, weights_only=True)
            except TypeError:
                state_dict = torch.load(stage1_model_path, map_location=dev)
            stage1_model.load_state_dict(state_dict)
            result["stage1_status"] = "ok"
        else:
            result["stage1_status"] = "model_missing"
            print("[transcribe_video] Notice: Running with initialized Stage 1 weights.")

        stage1_model.to(dev)
        stage1_model.eval()

        with torch.no_grad():
            logits, _ = stage1_model(video_tensor, video_lengths=video_lengths)
            decoded_phonemes = ML_VALLR.decode_ctc(logits, vocab)[0]

        result["phonemes"] = decoded_phonemes
        result["success"] = True
    except Exception as exc:
        result["error"] = f"Stage 1 inference error: {exc}"
        return result

    # Step 3: Stage 2 Phoneme-to-Devanagari Translation (Sarvam-1)
    try:
        translator = HindiPhonemeTranslator(adapter_path=sarvam_adapter_path, device=str(dev))
        if translator.is_mock or translator.model is None:
            result["stage2_status"] = "mock_mode"
            result["transcription"] = None
        else:
            result["transcription"] = translator.translate(decoded_phonemes)
            result["stage2_status"] = "ok"
    except Exception as exc:
        result["stage2_status"] = "failed"
        result["error"] = f"Stage 2 translation error: {exc}"

    return result

"""
Punarvaani Inference Interface
Backend-independent, production-ready inference API for Hindi Visual Speech Recognition.
Returns structured dictionaries with clear status indicators, error handling, and mock indicators.
"""

import os
import torch
from typing import Dict, Any, Optional

from Models.ML_VALLR import ML_VALLR
from Models.Sarvam import HindiPhonemeTranslator
from config import get_hindi_vocab


def load_inference_video(video_path: str, num_frames: int = 16, frame_size: tuple = (224, 224), crop_face: bool = True) -> Optional[torch.Tensor]:
    """Extracts, normalizes, and samples num_frames from the video file."""
    from Data.dataset import HindiVideoDataset
    dataset_helper = HindiVideoDataset.__new__(HindiVideoDataset)
    dataset_helper.num_frames = num_frames
    dataset_helper.frame_size = frame_size
    dataset_helper.crop_face = crop_face
    dataset_helper.face_cropper = None

    if crop_face:
        try:
            from face_cropper import FaceCropper
            dataset_helper.face_cropper = FaceCropper(
                min_face_detector_confidence=0.5,
                face_detector_model_selection="SHORT_RANGE",
                landmark_detector_static_image_mode="STATIC_MODE",
                min_landmark_detector_confidence=0.5
            )
        except Exception as e:
            print(f"[transcribe_video] Notice: FaceCropper unavailable ({e}). Using standard resize.")

    video_np = dataset_helper._load_video_frames(video_path)
    if video_np is None:
        return None

    # (T, H, W, C) -> (T, C, H, W) -> (1, T, C, H, W)
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0
    return video_tensor.unsqueeze(0)


def transcribe_video(
    video_path: str,
    stage1_model_path: Optional[str] = None,
    sarvam_adapter_path: Optional[str] = None,
    device: Optional[str] = None,
    crop_face: bool = True,
    temporal_upsample_factor: int = 4
) -> Dict[str, Any]:
    """
    Unified, framework-independent inference entry point for Hindi Visual Speech Recognition.

    Args:
        video_path: Path to the input video file (e.g. .mp4, .avi, .mkv).
        stage1_model_path: Path to Stage 1 ML_VALLR trained weights (.pth).
        sarvam_adapter_path: Path to directory containing trained Sarvam-1 LoRA weights.
        device: 'cuda', 'cpu', or None (auto-detect).
        crop_face: Whether to apply MediaPipe lip tracking.
        temporal_upsample_factor: Upsampling factor in temporal adapter (default: 4).

    Returns:
        Dict with keys:
            success (bool): True if at least Stage 1 phoneme prediction succeeded.
            phonemes (str): Decoded Hindi phoneme sequence.
            transcription (Optional[str]): Reconstructed Hindi Devanagari text, or None if mock/unavailable.
            stage1_status (str): "ok", "model_missing", "failed".
            stage2_status (str): "ok", "adapter_missing", "mock_mode", "failed".
            error (Optional[str]): Error message if execution encountered a fatal fault.
    """
    result = {
        "success": False,
        "phonemes": "",
        "transcription": None,
        "stage1_status": "failed",
        "stage2_status": "adapter_missing" if not sarvam_adapter_path else "failed",
        "error": None
    }

    if not video_path or not os.path.exists(video_path):
        result["error"] = f"Video file not found at: '{video_path}'"
        return result

    dev = torch.device(device if device and torch.cuda.is_available() and "cuda" in device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    vocab = get_hindi_vocab()

    # Step 1: Preprocessing & Frame Extraction
    try:
        video_tensor = load_inference_video(video_path, num_frames=16, crop_face=crop_face)
        if video_tensor is None:
            result["error"] = "Failed to decode video frames."
            return result
        video_tensor = video_tensor.to(dev)
    except Exception as e:
        result["error"] = f"Preprocessing error: {str(e)}"
        return result

    # Step 2: Stage 1 Visual-to-Phoneme Model
    try:
        stage1_model = ML_VALLR(
            adapter_dim=256,
            num_classes=len(vocab),
            temporal_upsample_factor=temporal_upsample_factor
        )
        if stage1_model_path and os.path.exists(stage1_model_path):
            state_dict = torch.load(stage1_model_path, map_location=dev)
            stage1_model.load_state_dict(state_dict)
            result["stage1_status"] = "ok"
        else:
            result["stage1_status"] = "model_missing"
            print(f"[transcribe_video] Notice: Running with initialized Stage 1 weights.")

        stage1_model.to(dev)
        stage1_model.eval()

        with torch.no_grad():
            logits, _ = stage1_model(video_tensor)
            decoded_phonemes = ML_VALLR.decode_ctc(logits, vocab)[0]

        result["phonemes"] = decoded_phonemes
        result["success"] = True
    except Exception as e:
        result["error"] = f"Stage 1 inference error: {str(e)}"
        return result

    # Step 3: Stage 2 Phoneme-to-Devanagari Translation (Sarvam-1)
    try:
        translator = HindiPhonemeTranslator(adapter_path=sarvam_adapter_path, device=str(dev))
        if translator.is_mock or translator.model is None:
            result["stage2_status"] = "mock_mode"
            result["transcription"] = None
        else:
            hindi_text = translator.translate(decoded_phonemes)
            result["transcription"] = hindi_text
            result["stage2_status"] = "ok"
    except Exception as e:
        result["stage2_status"] = "failed"
        result["error"] = f"Stage 2 translation error: {str(e)}"

    return result


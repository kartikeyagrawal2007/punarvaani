import os

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from Data.dataset import HindiVideoDataset, VSRBatch, custom_collate
from Data.metrics import corpus_cer_wer, levenshtein_distance
from Models.ML_VALLR import ML_VALLR
from Models.Sarvam import HindiPhonemeTranslator
from config import (
    get_hindi_vocab,
    get_vocab,
    load_args,
    parse_spatial_pool_tokens,
    resolve_crop_mode,
    WarmupScheduler,
)


# =============================================================================
# Shared helpers
# =============================================================================
def _amp_context(device: torch.device):
    """Autocast context plus a GradScaler only where it is needed."""
    if device.type == "cuda":
        return autocast(device_type="cuda", dtype=torch.float16), GradScaler("cuda")
    return autocast(device_type="cpu", dtype=torch.bfloat16), None


def _ctc_loss(criterion, logits, labels, input_lengths, device):
    """CTC in fp32 (autocast-safe), with per-sample input lengths."""
    log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1)  # (T, B, C)
    targets = torch.cat([label.to(device) for label in labels])
    target_lengths = torch.tensor(
        [len(label) for label in labels], dtype=torch.long, device=device
    )
    return criterion(log_probs, targets, input_lengths.to(device), target_lengths)


def _model_kwargs(args, num_classes: int):
    return dict(
        adapter_dim=256,
        num_classes=num_classes,
        temporal_upsample_factor=args.temporal_upsample_factor,
        feature_size=args.vit_embed_dim,
        vit_depth=args.vit_depth,
        vit_heads=args.vit_heads,
        spatial_pool_tokens=parse_spatial_pool_tokens(args.spatial_pool_tokens),
    )


def _load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # older torch
        return torch.load(path, map_location=device)


def load_videos(
    video_path: str,
    max_frames: int = 96,
    num_frames: int = 0,
    frame_size: tuple = (224, 224),
    crop_mode: str = "mouth",
    frame_stride: int = 1,
):
    """Load a clip as ``(1, T, C, H, W)`` plus its frame count."""
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
                "Install it with `pip install mediapipe` or pass --crop_mode none "
                "for an already-cropped video."
            ) from exc

    frames = dataset_helper._load_video_frames(video_path)
    if frames is None:
        return None, 0

    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    return tensor.unsqueeze(0), int(frames.shape[0])


def load_finetuned_model(model_path, device, num_classes, args=None, **model_kwargs):
    """Load Stage-1 weights with the architecture described by ``args``."""
    if args is not None:
        kwargs = _model_kwargs(args, num_classes)
        kwargs.update(model_kwargs)
    else:
        kwargs = dict(
            adapter_dim=256,
            num_classes=num_classes,
            temporal_upsample_factor=model_kwargs.pop("temporal_upsample_factor", 4),
            **model_kwargs,
        )

    model = ML_VALLR(**kwargs)
    if model_path and os.path.exists(model_path):
        state_dict = _load_state_dict(model_path, device)
        model.load_state_dict(state_dict)
        print(f"[Stage 1] Successfully loaded model weights from {model_path}")
    else:
        print(f"[Stage 1] Warning: No checkpoint found at '{model_path}'. Using initialized weights.")

    model.to(device)
    model.eval()
    return model


# =============================================================================
# Training
# =============================================================================
def train_one_epoch(model, dataloader, optimizer, criterion, device, phoneme_vocab):
    model.train()
    running_loss = 0.0
    total_edits = 0
    total_ref_phonemes = 0
    valid_batches = 0
    dropped = 0
    context, scaler = _amp_context(device)

    for batch in tqdm(dataloader, desc="Training", leave=False):
        if not isinstance(batch, VSRBatch):
            raise TypeError("custom_collate must return a VSRBatch")
        dropped += batch.dropped
        if batch.videos is None or len(batch) == 0 or not batch.labels:
            continue

        videos = batch.videos.to(device)
        video_lengths = batch.video_lengths.to(device)
        labels = [label.to(device) for label in batch.labels]

        optimizer.zero_grad(set_to_none=True)
        with context:
            raw_logits, _ = model(videos, video_lengths=video_lengths)
            loss = _ctc_loss(criterion, raw_logits, labels, batch.input_lengths, device)

        if not torch.isfinite(loss) or loss.item() < 0:
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        running_loss += loss.item()
        valid_batches += 1

        predicted_strings = ML_VALLR.decode_ctc(raw_logits.detach(), phoneme_vocab)
        reverse_vocab = {v: k for k, v in phoneme_vocab.items()}
        for pred_str, label_tensor in zip(predicted_strings, labels):
            true_str = " ".join(
                reverse_vocab.get(int(idx), "")
                for idx in label_tensor.detach().cpu()
                if int(idx) in reverse_vocab
            )
            total_edits += levenshtein_distance(pred_str.split(), true_str.split())
            total_ref_phonemes += len(true_str.split())

    avg_loss = running_loss / max(1, valid_batches)
    per = (total_edits / total_ref_phonemes * 100.0) if total_ref_phonemes else 100.0
    return avg_loss, per, dropped


def validate_one_epoch(model, dataloader, criterion, device, phoneme_vocab):
    model.eval()
    running_loss = 0.0
    total_edits = 0
    total_ref_phonemes = 0
    valid_batches = 0
    dropped = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            if not isinstance(batch, VSRBatch):
                raise TypeError("custom_collate must return a VSRBatch")
            dropped += batch.dropped
            if batch.videos is None or len(batch) == 0 or not batch.labels:
                continue

            videos = batch.videos.to(device)
            video_lengths = batch.video_lengths.to(device)
            labels = [label.to(device) for label in batch.labels]

            raw_logits, _ = model(videos, video_lengths=video_lengths)
            loss = _ctc_loss(criterion, raw_logits, labels, batch.input_lengths, device)
            if torch.isfinite(loss):
                running_loss += loss.item()
                valid_batches += 1

            predicted_strings = ML_VALLR.decode_ctc(raw_logits, phoneme_vocab)
            reverse_vocab = {v: k for k, v in phoneme_vocab.items()}
            for pred_str, label_tensor in zip(predicted_strings, labels):
                true_str = " ".join(
                    reverse_vocab.get(int(idx), "")
                    for idx in label_tensor.detach().cpu()
                    if int(idx) in reverse_vocab
                )
                total_edits += levenshtein_distance(pred_str.split(), true_str.split())
                total_ref_phonemes += len(true_str.split())

    avg_loss = running_loss / max(1, valid_batches)
    per = (total_edits / total_ref_phonemes * 100.0) if total_ref_phonemes else 100.0
    return avg_loss, per, dropped


def _make_dataset(args, vocab, split: str, data_dir: str, is_dedicated: bool):
    return HindiVideoDataset(
        data_dir=data_dir,
        phoneme_vocab=vocab,
        split=split,
        val_split=args.val_split,
        sample_size=args.sample_size,
        is_dedicated_split=is_dedicated,
        num_frames=args.num_frames,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        crop_mode=resolve_crop_mode(args),
        preprocessed=args.preprocessed,
        allow_unknown_speakers=args.allow_unknown_speakers,
        max_invalid_sample_pct=args.max_invalid_sample_pct,
        strict_ctc=args.strict_ctc,
        tubelet_size=2,
        temporal_upsample_factor=args.temporal_upsample_factor,
    )


def _loader(dataset, batch_size, shuffle, args, vocab):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=lambda batch: custom_collate(
            batch,
            vocab,
            tubelet_size=2,
            temporal_upsample_factor=args.temporal_upsample_factor,
        ),
    )


def train(args, device, vocab):
    data_dir = args.data_dir or args.videos_root
    if not data_dir:
        raise ValueError(
            "Please provide --data_dir pointing to the folder containing "
            "<speaker>/<clip>.mp4 and matching .txt files."
        )

    save_path = args.save_model_path or "./checkpoints/ml_vallr_hindi.pth"
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    print("\n==========================================")
    print(" Starting Training Stage 1: Video-to-Phoneme")
    print(f" Dataset Directory: {data_dir}")
    print(f" Target Classes: {len(vocab)} phonemes")
    print(f" Frames: {'all up to ' + str(args.max_frames) if not args.num_frames else args.num_frames}")
    print(f" Crop mode: {resolve_crop_mode(args)}")
    print(f" Checkpoint Target: {save_path}")
    print("==========================================\n")

    train_dataset = _make_dataset(args, vocab, "train", data_dir, is_dedicated=False)
    val_dataset = _make_dataset(
        args,
        vocab,
        "val",
        args.val_dir or data_dir,
        is_dedicated=args.val_dir is not None,
    )

    train_loader = _loader(train_dataset, args.batch_size, True, args, vocab)
    val_loader = _loader(val_dataset, args.eval_batch_size, False, args, vocab)

    model = ML_VALLR(**_model_kwargs(args, len(vocab))).to(device)
    criterion = nn.CTCLoss(blank=vocab["<pad>"], reduction="mean", zero_infinity=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = WarmupScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        base_lr=0.0,
        target_lr=1e-4,
        after_scheduler=ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3),
    )

    best_val_per = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_per, train_dropped = train_one_epoch(
            model, train_loader, optimizer, criterion, device, vocab
        )
        val_loss, val_per, val_dropped = validate_one_epoch(
            model, val_loader, criterion, device, vocab
        )
        scheduler.step(val_loss)
        lr = scheduler.get_last_lr()[0]

        dropped_limit = max(1, len(train_dataset)) * args.max_invalid_sample_pct / 100.0
        if train_dropped > dropped_limit:
            message = (
                f"{train_dropped} training samples were dropped this epoch "
                f"(limit {dropped_limit:.0f} at --max_invalid_sample_pct={args.max_invalid_sample_pct})."
            )
            if args.strict_ctc:
                raise RuntimeError(message)
            print(f"  [Warning] {message}")

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"| Train Loss: {train_loss:.4f} (PER: {train_per:.1f}%, dropped {train_dropped}) "
            f"| Val Loss: {val_loss:.4f} (PER: {val_per:.1f}%, dropped {val_dropped}) "
            f"| LR: {lr:.2e}"
        )

        if val_per < best_val_per:
            best_val_per = val_per
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best checkpoint to {save_path} (Val PER: {val_per:.1f}%)")


# =============================================================================
# Evaluation
# =============================================================================
def run_evaluation(args, device, vocab):
    eval_dir = args.val_dir or args.data_dir or args.videos_root
    if not eval_dir:
        raise ValueError("Please provide --data_dir or --val_dir for evaluation.")

    print("\n==========================================")
    print(f" Running Evaluation on: {eval_dir}")
    print(f" Stage 1 Checkpoint: {args.save_model_path}")
    print(f" Sarvam Adapter: {args.sarvam_adapter_path}")
    print("==========================================\n")

    eval_dataset = _make_dataset(args, vocab, "eval", eval_dir, is_dedicated=True)
    eval_loader = _loader(eval_dataset, args.eval_batch_size, False, args, vocab)

    model = load_finetuned_model(args.save_model_path, device, len(vocab), args=args)
    criterion = nn.CTCLoss(blank=vocab["<pad>"], reduction="mean", zero_infinity=True)

    val_loss, val_per, dropped = validate_one_epoch(
        model, eval_loader, criterion, device, vocab
    )
    print("\n--- Stage 1 Evaluation Results ---")
    print(f"  CTC Loss: {val_loss:.4f}")
    print(f"  Phoneme Error Rate (PER): {val_per:.2f}%")
    print(f"  Dropped samples (CTC length): {dropped}")

    if not args.sarvam_adapter_path:
        return

    translator = HindiPhonemeTranslator(adapter_path=args.sarvam_adapter_path, device=str(device))
    if translator.is_mock or translator.model is None:
        print("\n[Stage 2] Sarvam adapter is not active (MOCK mode) - skipping CER/WER.")
        print(f"[Stage 2] Reason: {getattr(translator, 'mock_reason', 'adapter unavailable')}")
        return

    predictions = []
    references = []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Stage 2 Evaluation"):
            if batch.videos is None or len(batch) == 0:
                continue
            videos = batch.videos.to(device)
            video_lengths = batch.video_lengths.to(device)
            raw_logits, _ = model(videos, video_lengths=video_lengths)
            pred_phonemes = ML_VALLR.decode_ctc(raw_logits, vocab)
            for phonemes, ref_text in zip(pred_phonemes, batch.texts):
                predictions.append(translator.translate(phonemes))
                references.append(ref_text)

    metrics = corpus_cer_wer(predictions, references)
    print("\n--- End-to-End Evaluation Results ---")
    print(f"  Character Error Rate (CER): {metrics['cer']:.2f}%")
    print(f"  Word Error Rate (WER): {metrics['wer']:.2f}%")


# =============================================================================
# Inference
# =============================================================================
def run_inference(
    video_path: str,
    stage1_model_path: str,
    sarvam_adapter_path: str,
    device: torch.device,
    vocab: dict,
    temporal_upsample_factor: int = 4,
    max_frames: int = 96,
    num_frames: int = 0,
    frame_stride: int = 1,
    crop_mode: str = "mouth",
    args=None,
):
    """Video -> ML_VALLR (CTC) -> phonemes -> Sarvam-1 -> Devanagari text."""
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"Inference video path '{video_path}' does not exist.")

    print(f"\n--- Running End-to-End Inference on: {video_path} ---")
    print("1. Extracting video frames & cropping...")
    video_tensor, num_frames_read = load_videos(
        video_path,
        max_frames=max_frames,
        num_frames=num_frames,
        crop_mode=crop_mode,
        frame_stride=frame_stride,
    )
    if video_tensor is None:
        print("Error: Could not extract frames from video.")
        return None
    video_tensor = video_tensor.to(device)
    video_lengths = torch.tensor([num_frames_read], dtype=torch.long, device=device)

    print("2. Predicting Hindi phonemes via Stage 1 (ML_VALLR)...")
    model = load_finetuned_model(stage1_model_path, device, len(vocab), args=args)
    with torch.no_grad():
        logits, _ = model(video_tensor, video_lengths=video_lengths)
        predicted_phonemes = ML_VALLR.decode_ctc(logits, vocab)[0]

    print(f"   -> Predicted Phonemes: '{predicted_phonemes}'")

    print("3. Reconstructing Devanagari Hindi text via Stage 2 (Sarvam-1)...")
    translator = HindiPhonemeTranslator(adapter_path=sarvam_adapter_path, device=str(device))
    hindi_transcript = translator.translate(predicted_phonemes)

    print("\n==========================================")
    print(f" PREDICTED PHONEMES : {predicted_phonemes}")
    print(f" FINAL HINDI TEXT   : {hindi_transcript}")
    print("==========================================\n")

    return {
        "phonemes": predicted_phonemes,
        "transcript": hindi_transcript,
        "num_frames": num_frames_read,
    }


def main():
    args = load_args()
    device = torch.device(
        args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    )
    vocab = get_hindi_vocab() if args.language == "hindi" else get_vocab("english")

    if args.mode == "train":
        train(args, device, vocab)
    elif args.mode == "eval":
        run_evaluation(args, device, vocab)
    elif args.mode == "infer":
        run_inference(
            video_path=args.infer_video_path,
            stage1_model_path=args.save_model_path,
            sarvam_adapter_path=args.sarvam_adapter_path,
            device=device,
            vocab=vocab,
            temporal_upsample_factor=args.temporal_upsample_factor,
            max_frames=args.max_frames,
            num_frames=args.num_frames,
            frame_stride=args.frame_stride,
            crop_mode=resolve_crop_mode(args),
            args=args,
        )


if __name__ == "__main__":
    main()

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from itertools import zip_longest
try:
    import jiwer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False


from Models.ML_VALLR import ML_VALLR, VALLR
from Models.Sarvam import HindiPhonemeTranslator
from Data.dataset import HindiVideoDataset, custom_collate
from config import load_args, WarmupScheduler, get_vocab, get_hindi_vocab


def load_videos(video_path: str, num_frames: int = 16, frame_size: tuple = (224, 224)):
    """Loads video frames from file path and returns tensor of shape (1, T, C, H, W)."""
    dataset_helper = HindiVideoDataset.__new__(HindiVideoDataset)
    dataset_helper.num_frames = num_frames
    dataset_helper.frame_size = frame_size
    dataset_helper.crop_face = True
    dataset_helper.face_cropper = None

    try:
        from face_cropper import FaceCropper
        dataset_helper.face_cropper = FaceCropper(
            min_face_detector_confidence=0.5,
            face_detector_model_selection="SHORT_RANGE",
            landmark_detector_static_image_mode="STATIC_MODE",
            min_landmark_detector_confidence=0.5
        )
    except Exception:
        pass

    video_np = dataset_helper._load_video_frames(video_path)
    if video_np is None:
        return None

    # (T, H, W, C) -> (T, C, H, W) -> (1, T, C, H, W)
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0
    return video_tensor.unsqueeze(0)


def load_finetuned_model(model_path: str, device: torch.device, num_classes: int):
    """Loads Stage 1 ML_VALLR model weights from checkpoint."""
    model = ML_VALLR(
        adapter_dim=256,
        num_classes=num_classes
    )
    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[Stage 1] Successfully loaded model weights from {model_path}")
    else:
        print(f"[Stage 1] Warning: No checkpoint found at '{model_path}'. Running with initialized weights.")

    model.to(device)
    model.eval()
    return model


def train_one_epoch(model, dataloader, optimizer, criterion, device, phoneme_vocab):
    """Trains Stage 1 model for one epoch using mixed precision CTC Loss."""
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    scaler = GradScaler()

    for batch in tqdm(dataloader, desc="Training", leave=False):
        videos, labels, _ = batch
        if videos.size(0) == 0 or len(labels) == 0:
            continue

        videos = videos.to(device)
        labels = [l.to(device) for l in labels]

        optimizer.zero_grad()

        with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            raw_logits, _ = model(videos)  # (B, T_down, num_classes)
            log_probs = raw_logits.log_softmax(dim=-1).transpose(0, 1)  # (T_down, B, num_classes)

            batch_size = log_probs.size(1)
            time_steps = log_probs.size(0)
            input_lengths = torch.full(size=(batch_size,), fill_value=time_steps, dtype=torch.long, device=device)
            target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long, device=device)

            # Skip batch if CTC length constraint is violated
            if input_lengths.min() < target_lengths.max():
                continue

            targets = torch.cat(labels)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            if torch.isnan(loss) or loss.item() < 0:
                continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # Compute phoneme accuracy
        predicted_strings = ML_VALLR.decode_ctc(raw_logits, phoneme_vocab)
        reverse_vocab = {v: k for k, v in phoneme_vocab.items()}
        for pred_str, label_tensor in zip(predicted_strings, labels):
            true_str = " ".join([reverse_vocab.get(idx.item(), "") for idx in label_tensor if idx.item() in reverse_vocab])
            pred_tokens = pred_str.split()
            true_tokens = true_str.split()
            correct = sum(p == t for p, t in zip(pred_tokens, true_tokens))
            total_correct += correct
            total_samples += len(true_tokens)

    avg_loss = running_loss / max(1, len(dataloader))
    accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
    return avg_loss, accuracy


def validate_one_epoch(model, dataloader, criterion, device, phoneme_vocab):
    """Evaluates Stage 1 model on validation dataloader."""
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            videos, labels, _ = batch
            if videos.size(0) == 0 or len(labels) == 0:
                continue

            videos = videos.to(device)
            labels = [l.to(device) for l in labels]

            raw_logits, _ = model(videos)
            log_probs = raw_logits.log_softmax(dim=-1).transpose(0, 1)

            batch_size = log_probs.size(1)
            time_steps = log_probs.size(0)
            input_lengths = torch.full(size=(batch_size,), fill_value=time_steps, dtype=torch.long, device=device)
            target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long, device=device)

            if input_lengths.min() < target_lengths.max():
                continue

            loss = criterion(log_probs, torch.cat(labels), input_lengths, target_lengths)
            if not torch.isnan(loss):
                running_loss += loss.item()

            predicted_strings = ML_VALLR.decode_ctc(raw_logits, phoneme_vocab)
            reverse_vocab = {v: k for k, v in phoneme_vocab.items()}
            for pred_str, label_tensor in zip(predicted_strings, labels):
                true_str = " ".join([reverse_vocab.get(idx.item(), "") for idx in label_tensor if idx.item() in reverse_vocab])
                pred_tokens = pred_str.split()
                true_tokens = true_str.split()
                correct = sum(p == t for p, t in zip(pred_tokens, true_tokens))
                total_correct += correct
                total_samples += len(true_tokens)

    avg_loss = running_loss / max(1, len(dataloader))
    accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
    return avg_loss, accuracy


def train(args, device, vocab):
    """Main training orchestration for Stage 1 on custom Hindi video dataset."""
    data_dir = args.data_dir or args.videos_root
    if not data_dir:
        raise ValueError("Please provide --data_dir pointing to the folder containing dataXXX.mp4 and dataXXX.txt files.")

    save_path = args.save_model_path or "./checkpoints/ml_vallr_hindi.pth"
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    print(f"\n==========================================")
    print(f" Starting Training Stage 1: Video-to-Phoneme")
    print(f" Dataset Directory: {data_dir}")
    print(f" Target Classes: {len(vocab)} phonemes")
    print(f" Checkpoint Target: {save_path}")
    print(f"==========================================\n")

    train_dataset = HindiVideoDataset(
        data_dir=data_dir,
        phoneme_vocab=vocab,
        split="train",
        val_split=args.val_split,
        num_frames=args.num_frames,
        crop_face=args.crop_face
    )

    val_dataset = HindiVideoDataset(
        data_dir=args.val_dir or data_dir,
        phoneme_vocab=vocab,
        split="val",
        val_split=args.val_split,
        num_frames=args.num_frames,
        crop_face=args.crop_face
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: custom_collate(b, vocab)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: custom_collate(b, vocab)
    )

    model = ML_VALLR(adapter_dim=256, num_classes=len(vocab)).to(device)
    criterion = nn.CTCLoss(blank=vocab['<pad>'], reduction="mean", zero_infinity=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, vocab)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device, vocab)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch}/{args.epochs}] "
              f"| Train Loss: {train_loss:.4f} (Acc: {train_acc:.1f}%) "
              f"| Val Loss: {val_loss:.4f} (Acc: {val_acc:.1f}%) "
              f"| LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best checkpoint to {save_path}")


def run_inference(video_path: str, stage1_model_path: str, sarvam_adapter_path: str, device: torch.device, vocab: dict):
    """
    Complete Two-Stage End-to-End Inference:
    Video -> ML_VALLR (CTC) -> Phoneme String -> Sarvam-1 Indic LLM -> Devanagari Hindi Text
    """
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"Inference video path '{video_path}' does not exist.")

    print(f"\n--- Running End-to-End Inference on: {video_path} ---")

    # Step 1: Load and preprocess video frames
    print("1. Extracting video frames & applying face cropping...")
    video_tensor = load_videos(video_path, num_frames=16)
    if video_tensor is None:
        print("Error: Could not extract frames from video.")
        return

    # Step 2: Stage 1 CTC Inference
    print("2. Predicting Hindi phonemes via Stage 1 (ML_VALLR)...")
    stage1_model = load_finetuned_model(stage1_model_path, device, num_classes=len(vocab))
    video_tensor = video_tensor.to(device)

    with torch.no_grad():
        logits, _ = stage1_model(video_tensor)
        predicted_phonemes = ML_VALLR.decode_ctc(logits, vocab)[0]

    print(f"   -> Predicted Phonemes: '{predicted_phonemes}'")

    # Step 3: Stage 2 Phoneme-to-Sentence Translation (Sarvam-1)
    print("3. Reconstructing Devanagari Hindi text via Stage 2 (Sarvam-1)...")
    translator = HindiPhonemeTranslator(adapter_path=sarvam_adapter_path, device=device)
    hindi_transcript = translator.translate(predicted_phonemes)

    print(f"\n==========================================")
    print(f" PREDICTED PHONEMES : {predicted_phonemes}")
    print(f" FINAL HINDI TEXT   : {hindi_transcript}")
    print(f"==========================================\n")

    return {
        "phonemes": predicted_phonemes,
        "transcript": hindi_transcript
    }


def main():
    args = load_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vocab = get_hindi_vocab() if args.language == "hindi" else get_vocab("english")

    if args.mode == "train":
        train(args, device, vocab)
    elif args.mode == "infer":
        run_inference(
            video_path=args.infer_video_path,
            stage1_model_path=args.save_model_path,
            sarvam_adapter_path=args.sarvam_adapter_path,
            device=device,
            vocab=vocab
        )


if __name__ == "__main__":
    main()

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


def levenshtein_distance(seq1, seq2):
    """Computes Levenshtein edit distance between two sequences (tokens or characters)."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def load_finetuned_model(model_path: str, device: torch.device, num_classes: int, temporal_upsample_factor: int = 4):
    """Loads Stage 1 ML_VALLR model weights from checkpoint."""
    model = ML_VALLR(
        adapter_dim=256,
        num_classes=num_classes,
        temporal_upsample_factor=temporal_upsample_factor
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
    total_edits = 0
    total_ref_phonemes = 0
    scaler = GradScaler()
    valid_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        if batch is None or batch[0] is None or len(batch[1]) == 0:
            continue

        videos, labels, _ = batch
        if videos.size(0) == 0:
            continue

        videos = videos.to(device)
        labels = [l.to(device) for l in labels]

        optimizer.zero_grad()

        with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            raw_logits, _ = model(videos)  # (B, T_ctc, num_classes)
            log_probs = raw_logits.log_softmax(dim=-1).transpose(0, 1)  # (T_ctc, B, num_classes)

            batch_size = log_probs.size(1)
            time_steps = log_probs.size(0)
            input_lengths = torch.full(size=(batch_size,), fill_value=time_steps, dtype=torch.long, device=device)
            target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long, device=device)

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
        valid_batches += 1

        # Compute Phoneme Error Rate (PER) via Levenshtein edit distance
        predicted_strings = ML_VALLR.decode_ctc(raw_logits, phoneme_vocab)
        reverse_vocab = {v: k for k, v in phoneme_vocab.items()}
        for pred_str, label_tensor in zip(predicted_strings, labels):
            true_str = " ".join([reverse_vocab.get(idx.item(), "") for idx in label_tensor if idx.item() in reverse_vocab])
            edits = levenshtein_distance(pred_str.split(), true_str.split())
            total_edits += edits
            total_ref_phonemes += len(true_str.split())

    avg_loss = running_loss / max(1, valid_batches)
    per = (total_edits / total_ref_phonemes * 100) if total_ref_phonemes > 0 else 100.0
    return avg_loss, per


def validate_one_epoch(model, dataloader, criterion, device, phoneme_vocab):
    """Evaluates Stage 1 model on validation dataloader."""
    model.eval()
    running_loss = 0.0
    total_edits = 0
    total_ref_phonemes = 0
    valid_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            if batch is None or batch[0] is None or len(batch[1]) == 0:
                continue

            videos, labels, _ = batch
            if videos.size(0) == 0:
                continue

            videos = videos.to(device)
            labels = [l.to(device) for l in labels]

            raw_logits, _ = model(videos)
            log_probs = raw_logits.log_softmax(dim=-1).transpose(0, 1)

            batch_size = log_probs.size(1)
            time_steps = log_probs.size(0)
            input_lengths = torch.full(size=(batch_size,), fill_value=time_steps, dtype=torch.long, device=device)
            target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long, device=device)

            loss = criterion(log_probs, torch.cat(labels), input_lengths, target_lengths)
            if not torch.isnan(loss):
                running_loss += loss.item()
                valid_batches += 1

            predicted_strings = ML_VALLR.decode_ctc(raw_logits, phoneme_vocab)
            reverse_vocab = {v: k for k, v in phoneme_vocab.items()}
            for pred_str, label_tensor in zip(predicted_strings, labels):
                true_str = " ".join([reverse_vocab.get(idx.item(), "") for idx in label_tensor if idx.item() in reverse_vocab])
                edits = levenshtein_distance(pred_str.split(), true_str.split())
                total_edits += edits
                total_ref_phonemes += len(true_str.split())

    avg_loss = running_loss / max(1, valid_batches)
    per = (total_edits / total_ref_phonemes * 100) if total_ref_phonemes > 0 else 100.0
    return avg_loss, per


def train(args, device, vocab):
    """Main training orchestration for Stage 1 on custom Hindi video dataset."""
    data_dir = args.data_dir or args.videos_root
    if not data_dir:
        raise ValueError("Please provide --data_dir pointing to the folder containing dataXXX.mp4 and dataXXX.txt files.")

    save_path = args.save_model_path or "./checkpoints/ml_vallr_hindi.pth"
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    t_ctc = (args.num_frames // 2) * args.temporal_upsample_factor

    print(f"\n==========================================")
    print(f" Starting Training Stage 1: Video-to-Phoneme")
    print(f" Dataset Directory: {data_dir}")
    print(f" Target Classes: {len(vocab)} phonemes")
    print(f" Frames: {args.num_frames} -> Tubelets: {args.num_frames // 2} -> T_ctc: {t_ctc}")
    print(f" Checkpoint Target: {save_path}")
    print(f"==========================================\n")

    train_dataset = HindiVideoDataset(
        data_dir=data_dir,
        phoneme_vocab=vocab,
        split="train",
        val_split=args.val_split,
        sample_size=args.sample_size,
        num_frames=args.num_frames,
        crop_face=args.crop_face
    )

    val_dataset = HindiVideoDataset(
        data_dir=args.val_dir or data_dir,
        phoneme_vocab=vocab,
        split="val",
        val_split=args.val_split,
        sample_size=args.sample_size,
        is_dedicated_split=(args.val_dir is not None),
        num_frames=args.num_frames,
        crop_face=args.crop_face
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: custom_collate(b, vocab, t_ctc=t_ctc)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: custom_collate(b, vocab, t_ctc=t_ctc)
    )

    model = ML_VALLR(
        adapter_dim=256,
        num_classes=len(vocab),
        temporal_upsample_factor=args.temporal_upsample_factor
    ).to(device)
    criterion = nn.CTCLoss(blank=vocab['<pad>'], reduction="mean", zero_infinity=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_per = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_per = train_one_epoch(model, train_loader, optimizer, criterion, device, vocab)
        val_loss, val_per = validate_one_epoch(model, val_loader, criterion, device, vocab)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch}/{args.epochs}] "
              f"| Train Loss: {train_loss:.4f} (PER: {train_per:.1f}%) "
              f"| Val Loss: {val_loss:.4f} (PER: {val_per:.1f}%) "
              f"| LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save checkpoint based on best validation PER
        if val_per < best_val_per:
            best_val_per = val_per
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best checkpoint to {save_path} (Val PER: {val_per:.1f}%)")


def run_evaluation(args, device, vocab):
    """Evaluates fine-tuned model on validation/test set, computing PER, CER, and WER."""
    eval_dir = args.val_dir or args.data_dir or args.videos_root
    if not eval_dir:
        raise ValueError("Please provide --data_dir or --val_dir pointing to the evaluation dataset.")

    t_ctc = (args.num_frames // 2) * args.temporal_upsample_factor
    print(f"\n==========================================")
    print(f" Running Evaluation on: {eval_dir}")
    print(f" Stage 1 Checkpoint: {args.save_model_path}")
    print(f" Sarvam Adapter: {args.sarvam_adapter_path}")
    print(f"==========================================\n")

    eval_dataset = HindiVideoDataset(
        data_dir=eval_dir,
        phoneme_vocab=vocab,
        split="eval",
        val_split=0.0,
        sample_size=args.sample_size,
        is_dedicated_split=True,
        num_frames=args.num_frames,
        crop_face=args.crop_face
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: custom_collate(b, vocab, t_ctc=t_ctc)
    )

    model = load_finetuned_model(
        args.save_model_path,
        device,
        num_classes=len(vocab),
        temporal_upsample_factor=args.temporal_upsample_factor
    )
    criterion = nn.CTCLoss(blank=vocab['<pad>'], reduction="mean", zero_infinity=True)

    val_loss, val_per = validate_one_epoch(model, eval_loader, criterion, device, vocab)
    print(f"\n--- Stage 1 Evaluation Results ---")
    print(f"  CTC Loss: {val_loss:.4f}")
    print(f"  Phoneme Error Rate (PER): {val_per:.2f}%")

    if args.sarvam_adapter_path:
        translator = HindiPhonemeTranslator(adapter_path=args.sarvam_adapter_path, device=device)
        total_char_edits = 0
        total_ref_chars = 0
        total_word_edits = 0
        total_ref_words = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Stage 2 Evaluation"):
                if batch is None or batch[0] is None:
                    continue
                videos, labels, texts = batch
                videos = videos.to(device)
                raw_logits, _ = model(videos)
                pred_phonemes_batch = ML_VALLR.decode_ctc(raw_logits, vocab)

                for pred_phon, ref_text in zip(pred_phonemes_batch, texts):
                    pred_hindi = translator.translate(pred_phon)
                    # Normalize texts
                    ref_clean = re.sub(r'[^\u0900-\u097F\s]', '', ref_text).strip()
                    pred_clean = re.sub(r'[^\u0900-\u097F\s]', '', pred_hindi).strip()

                    # CER
                    total_char_edits += levenshtein_distance(list(pred_clean), list(ref_clean))
                    total_ref_chars += len(list(ref_clean))

                    # WER
                    total_word_edits += levenshtein_distance(pred_clean.split(), ref_clean.split())
                    total_ref_words += len(ref_clean.split())

        cer = (total_char_edits / total_ref_chars * 100) if total_ref_chars > 0 else 100.0
        wer = (total_word_edits / total_ref_words * 100) if total_ref_words > 0 else 100.0
        print(f"\n--- End-to-End Evaluation Results ---")
        print(f"  Character Error Rate (CER): {cer:.2f}%")
        print(f"  Word Error Rate (WER): {wer:.2f}%")


def run_inference(video_path: str, stage1_model_path: str, sarvam_adapter_path: str, device: torch.device, vocab: dict, temporal_upsample_factor: int = 4):
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
    stage1_model = load_finetuned_model(
        stage1_model_path,
        device,
        num_classes=len(vocab),
        temporal_upsample_factor=temporal_upsample_factor
    )
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
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
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
            temporal_upsample_factor=args.temporal_upsample_factor
        )


if __name__ == "__main__":
    main()

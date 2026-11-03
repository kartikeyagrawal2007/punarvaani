try:
    import configargparse as argparse_lib
except ImportError:
    import argparse as argparse_lib

import csv
import numpy as np

def load_args():
    parser = argparse_lib.ArgumentParser(description="Main")


    # Device configuration
    parser.add_argument('--device', type=str, default='cuda', help="Device to use for computation ('cuda' or 'cpu')")

    # Model configuration
    # parser.add_argument('--ckpt_path', default=None, help="Path to the model checkpoint for loading")
    parser.add_argument('--save_model_path', type=str, help="Path to save the trained model")
    # parser.add_argument('--model_path', type=str, help="Path to the trained model for inference")
    parser.add_argument('--version', type=str, choices=['V1', 'V2'], default='V1', help="Model version to use: 'V1' or 'V2'")

    # Data configuration
    # parser.add_argument('--feat_dim', type=int, default=512, help="Video features dimension (used if loading features directly)")
    parser.add_argument('--videos_root', type=str, help="Root directory of the video dataset")
    # parser.add_argument('--audio_root', type=str, help="Root directory of the audio dataset")
    # parser.add_argument('--file_list', type=str, help="(List of) video file paths relative to videos_root, can also be regex")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training (reduced for all-frames input)")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of worker threads for loading data")
    # parser.add_argument('--whisper_model', type=str, default="base", help="Size of the whisper model to use for transcription")

    # Transformer configuration
    # parser.add_argument('--num_blocks', type=int, default=6, help="Number of transformer blocks")
    # parser.add_argument('--hidden_units', type=int, default=512, help="Transformer model size (hidden units)")
    # parser.add_argument('--num_heads', type=int, default=8, help="Number of attention heads")
    # parser.add_argument('--dropout_rate', type=float, default=0.1, help="Dropout probability in the transformer")

    # Preprocessing
    # parser.add_argument('--frame_size', type=int, default=224, help="Resize the input video frames to this resolution")
    # parser.add_argument('--videos_output', type=str, help="Output directory for the preporcessed video dataset")

    # Custom dataset and Hindi VSR configuration
    parser.add_argument('--data_dir', type=str, default=None, help="Root directory containing paired video files (dataXXX.mp4) and transcripts (dataXXX.txt)")
    parser.add_argument('--val_dir', type=str, default=None, help="Optional separate directory for validation data")
    parser.add_argument('--val_split', type=float, default=0.1, help="Fraction of data used for validation when using a single data_dir")
    parser.add_argument('--num_frames', type=int, default=0, help="Uniformly sample this many frames per clip. 0 = use all frames (up to --max_frames)")
    parser.add_argument('--max_frames', type=int, default=96, help="Maximum frames read per clip when --num_frames 0 (96 ~= 3.8s at 25fps)")
    parser.add_argument('--frame_stride', type=int, default=1, help="Read every Nth frame from the source video")
    parser.add_argument('--infer_video_path', type=str, default=None, help="Path to video file for inference")
    parser.add_argument('--language', type=str, choices=['hindi', 'english'], default='hindi', help="Target language vocabulary ('hindi' or 'english')")
    parser.add_argument('--crop_face', dest='crop_face', action='store_true', help="Enable MediaPipe mouth/face cropping")
    parser.add_argument('--no_crop_face', dest='crop_face', action='store_false', help="Disable mouth/face cropping (use the clip as-is)")
    parser.add_argument('--crop_mode', type=str, choices=['mouth', 'face', 'none'], default=None, help="Crop strategy. Defaults to 'mouth' when cropping is enabled, else 'none'")
    parser.add_argument('--preprocessed', action='store_true', help="Clips are already mouth-cropped; skip cropping")
    parser.add_argument('--allow_unknown_speakers', action='store_true', help="Allow clips directly under data_dir (random clip split instead of speaker-disjoint)")
    parser.set_defaults(crop_face=True)
    parser.add_argument('--strict_ctc', action='store_true', help="Raise instead of warning when too many samples violate the CTC length constraint")

    # Architectural parameters
    parser.add_argument('--temporal_upsample_factor', type=int, default=4, help="Learned temporal upsampling factor in TemporalAdapter")
    parser.add_argument('--max_invalid_sample_pct', type=float, default=10.0, help="Maximum allowed percentage of invalid CTC samples per epoch")
    parser.add_argument('--eval_batch_size', type=int, default=4, help="Batch size for validation and evaluation mode")
    parser.add_argument('--warmup_steps', type=int, default=2000, help="Linear LR warmup steps before ReduceLROnPlateau")
    parser.add_argument('--vit_embed_dim', type=int, default=768, help="Spatiotemporal ViT embedding dimension")
    parser.add_argument('--vit_depth', type=int, default=8, help="Number of ViT blocks")
    parser.add_argument('--vit_heads', type=int, default=12, help="Number of ViT attention heads")
    parser.add_argument('--spatial_pool_tokens', type=str, default='7x7', help="Pool ViT spatial grid to HxW before attention, or 'none'")

    # Stage 2 LLM (Sarvam-1) configuration
    parser.add_argument('--sarvam_adapter_path', type=str, default=None, help="Path to directory containing fine-tuned Sarvam-1 LoRA adapter weights")
    parser.add_argument('--sarvam_base_model', type=str, default="sarvamai/sarvam-1", help="Hugging Face base model ID for Sarvam")

    # Training and inference parameters
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--sample_size', type=float, default=1.0, help="Percentage of the dataset to use for training")
    parser.add_argument('--mode', type=str, choices=['train', 'infer', 'eval'], default='train', help="Mode to run: 'train', 'infer' or 'eval'")

    args = parser.parse_args()

    return args

def parse_spatial_pool_tokens(value):
    """Parse 'HxW', 'N', or 'none' into a (H, W) tuple or None."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "none", "0"):
        return None
    if "x" in text:
        height, width = text.split("x", 1)
        return (int(height), int(width))
    size = int(text)
    return (size, size)


def resolve_crop_mode(args):
    """Resolve the effective crop mode from the CLI flags."""
    if getattr(args, "preprocessed", False):
        return "none"
    explicit = getattr(args, "crop_mode", None)
    if explicit:
        return explicit
    return "mouth" if getattr(args, "crop_face", True) else "none"


def get_hindi_vocab():
    """
    Standard 55-class Hindi Phoneme Inventory (HPI) for Visual Speech Recognition.
    0: <pad> (CTC blank token)
    1: | (Word boundary delimiter)
    2-54: Hindi vowels, consonants, flaps, loans.
    """
    hindi_vocab = {
        '<pad>': 0,     # CTC blank token
        '|': 1,         # Word boundary delimiter
        # Vowels
        'A': 2, 'AA': 3, 'I': 4, 'II': 5, 'U': 6, 'UU': 7,
        'E': 8, 'AI': 9, 'O': 10, 'AU': 11, 'RI': 12, 'AN': 13, 'AH': 14,
        # Velar
        'K': 15, 'KH': 16, 'G': 17, 'GH': 18, 'NG': 19,
        # Palatal
        'CH': 20, 'CHH': 21, 'J': 22, 'JH': 23, 'NY': 24,
        # Retroflex
        'TR': 25, 'THR': 26, 'DR': 27, 'DHR': 28, 'NR': 29,
        'RD': 30, 'RHD': 31, # Flaps: ड़, ढ़
        # Dental
        'TD': 32, 'THD': 33, 'DD': 34, 'DHD': 35, 'N': 36,
        # Labial
        'P': 37, 'PH': 38, 'B': 39, 'BH': 40, 'M': 41,
        # Approximants / Semivowels
        'Y': 42, 'R': 43, 'L': 44, 'V': 45,
        # Sibilants / Fricatives
        'SH': 46, 'SHR': 47, 'S': 48, 'H': 49,
        # Loan / Nuqta Consonants
        'Q': 50, 'KH_LOAN': 51, 'GH_LOAN': 52, 'Z': 53, 'F': 54
    }
    return hindi_vocab

def get_vocab(language="hindi"):
    if language == "hindi":
        return get_hindi_vocab()

    # Legacy English ARPAbet vocab
    phoneme_vocab = {
        '<pad>': 0, # Blank token for CTC loss       
        'AA': 1, 'AE': 2, 'AH': 3, 'AO': 4, 'AW': 5, 'AY': 6,
        'B': 7, 'CH': 8, 'D': 9, 'DH': 10, 'EH': 11, 'ER': 12,
        'EY': 13, 'F': 14, 'G': 15, 'HH': 16, 'IH': 17, 'IY': 18,
        'JH': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'NG': 24,
        'OW': 25, 'OY': 26, 'P': 27, 'R': 28, 'S': 29, 'SH': 30,
        'T': 31, 'TH': 32, 'UH': 33, 'UW': 34, 'V': 35, 'W': 36,
        'Y': 37, 'Z': 38, 'ZH': 39
    }
    return phoneme_vocab


class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, base_lr, target_lr, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler
        self.current_step = 0
        self._last_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, loss=None):
        self.current_step += 1
        
        # During warmup
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr + (self.target_lr - self.base_lr) * (self.current_step / self.warmup_steps)
            self._set_lr(lr)
        else:
            # After warmup, defer to the after_scheduler (if any)
            if self.after_scheduler is not None:
                if loss is not None:
                    self.after_scheduler.step(loss)
                else:
                    self.after_scheduler.step()
            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self._last_lr = [lr for _ in self.optimizer.param_groups]  # Update last_lr

    def get_last_lr(self):
        return self._last_lr
    
def log_ctc_inputs_to_csv(logits, labels, input_lengths, target_lengths, filepath="ctc_loss_inputs.csv"):
    """
    Log the values of logits, labels, input lengths, and target lengths to a CSV file.
    """
    # Flatten the logits and labels for easier CSV logging
    logits_flat = logits.detach().cpu().numpy().flatten()
    labels_flat = labels.detach().cpu().numpy().flatten()
    input_lengths_flat = input_lengths.detach().cpu().numpy().flatten()
    target_lengths_flat = target_lengths.detach().cpu().numpy().flatten()

    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Logits"] + list(logits_flat))
        writer.writerow(["Labels"] + list(labels_flat))
        writer.writerow(["Input Lengths"] + list(input_lengths_flat))
        writer.writerow(["Target Lengths"] + list(target_lengths_flat))
        writer.writerow([])  # Add a blank row for separation

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    """Generate sampled frame indices for video."""
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

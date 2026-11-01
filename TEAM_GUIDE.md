# Punarvaani (पुनर्वाणी) — Teammate Collaboration & Handover Guide

This guide provides concrete, step-by-step instructions for teammates responsible for:
1. **Teammate 1 (Phase 2):** Hindi Video Dataset Collection, Normalization & Face Preprocessing.
2. **Teammate 2 (Phase 3):** Sarvam-1 Indic LLM Fine-Tuning, Noise Modeling & LoRA Handover.

---

## 🏛️ How the Two Parts Connect to Core

```
┌─────────────────────────────────┐
│        TEAMMATE 1               │
│  Dataset & Video Preprocessing  │
│  (Raw Video ──► Cropped Lips)   │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     CORE STABILIZED PIPELINE    │
│  Stage 1: Spatiotemporal ViT    │
│  (Lip Movements ──► Phonemes)   │
└────────────────┬────────────────┘
                 │
                 ▼ [Predicted Phoneme Stream]
┌─────────────────────────────────┐
│        TEAMMATE 2               │
│  Stage 2: Fine-Tuned Sarvam-1   │
│  (Phonemes ──► Devanagari Text) │
└─────────────────────────────────┘
```

---

# 📘 Part 1: Guide for Teammate 1 (Dataset & Video Preprocessing)

### 1. Video Recording Specifications
* **Angle**: Direct frontal angle ($0^\circ$ pitch and yaw). The speaker must look straight into the camera.
* **Lighting**: Soft, diffused lighting across the whole face. Avoid downward overhead lighting that casts dark shadows under the lower lip.
* **Framing**: Face centered, mouth completely unobstructed. No microphones, hands, scarves, or hair obscuring the lips.
* **Resolution & FPS**: Minimum 720p at 25 or 30 frames per second.
* **Clip Length (Crucial for CTC Alignment)**:
  * **Target Duration**: **1.5 to 2.5 seconds per clip** (approximately 2 to 6 words, 10 to 25 syllables).
  * ⚠️ *Why this matters*: The Stage 1 model samples 16 frames and expands them to 32 CTC timesteps. The CTC alignment condition strictly requires $T \ge U + R$ (where $U$ is target phoneme count and $R$ is repeated letters). Long clips (> 3 seconds) with 40+ phonemes will be rejected during training.
* **Dataset Target Size**:
  * Minimum prototype: 300–500 clips across 5–10 distinct speakers.
  * Research benchmark: 1,500+ clips.

### 2. Transcript Formatting & Normalization
Pair each video with an exact matching text file:
* `spk01_clip001.mp4`
* `spk01_clip001.txt` (UTF-8 encoded)

**Rules for Transcripts:**
1. **Pure Devanagari Only**: Do not mix English alphabets (`A-Z`) or symbols into the `.txt` file.
2. **Spell Out Numbers**:
   - ❌ `२०२५ में ५ लोग थे` or `2025 me 5 log the`
   - ✅ `दो हज़ार पच्चीस में पाँच लोग थे`
3. **Transliterate English Loanwords**:
   - ❌ `School bus`
   - ✅ `स्कूल बस`
4. **Remove Punctuation**: Strip commas, hyphens, quotes, question marks, and emojis. Only spaces and full stops/dandas (`।`) are permitted.

### 3. Folder Organization & Preventing Speaker Leakage
Arrange the raw data by speaker so the validation split evaluates generalization to unseen faces:

```text
raw_videos/
├── spk01/
│   ├── spk01_001.mp4
│   ├── spk01_001.txt
│   └── spk01_002.mp4
├── spk02/
│   ├── spk02_001.mp4
│   └── spk02_001.txt
```

### 4. Running the Offline MediaPipe Cropper
Run the provided batch cropper to detect lips, smooth landmark jitter, and output normalized $224 \times 224$ crops:

```bash
python scripts/preprocess_lips.py \
  --input_dir ./raw_videos \
  --output_dir ./hindi_dataset \
  --num_frames 16 \
  --frame_size 224
```

### 5. Teammate 1 Verification Checklist
- [ ] Preprocessing crop success rate is $\ge 95\%$.
- [ ] No dropped frames or corrupted video containers.
- [ ] All transcripts parse cleanly through `python -m unittest tests/test_punarvaani.py`.
- [ ] Zero speaker overlap between train and validation splits.

---

# 📗 Part 2: Guide for Teammate 2 (Sarvam-1 LoRA Fine-Tuning)

### 1. Role of Stage 2 (Viseme Disambiguation)
Visual speech recognition suffers from **viseme ambiguity**—different phonemes look identical on the mouth:
* **/P/**, **/B/**, and **/M/** are all bilabial closures. Lip-reading cannot visually tell "बात" (*baat*) from "पात" (*paat*).
* Sarvam-1 acts as the linguistic context engine that converts noisy, ambiguous phonemes into grammatically coherent Devanagari text.

### 2. The S2S Special Token Contract
The model must be trained using this **exact prompt structure**:

```text
<S2S>
<PHONEMES>
{phoneme_sequence}
</PHONEMES>
<TEXT>
{devanagari_transcription}
```

**Required Special Tokens:**
```python
TAGS = ["<S2S>", "<PHONEMES>", "</PHONEMES>", "<TEXT>"]
```

⚠️ **Critical Tokenizer Rule:**
Before training, register the tags and resize embeddings:
```python
tokenizer.add_special_tokens({"additional_special_tokens": TAGS})
model.resize_token_embeddings(len(tokenizer))
```
Verify that `len(tokenizer.encode("<S2S>", add_special_tokens=False)) == 1`. Each tag must encode as a single token, never split into subwords.

### 3. Generating Large-Scale Training Data with Synthetic Noise
Teammate 2 does **not** need to wait for video recordings. You can train Sarvam-1 immediately on hundreds of thousands of sentences from any Hindi text corpus (e.g., AI4Bharat IndicCorp, Wikipedia, or OSCAR):

1. **Convert Text to Phonemes:**
   ```python
   from Data.hindi_g2p import devanagari_to_phonemes

   clean_text = "नमस्ते भारत"
   phonemes = devanagari_to_phonemes(clean_text, as_string=True)
   # 'N M S TD E | BH AA R TD'
   ```

2. **Inject Synthetic Visual Speech Noise (Crucial!):**
   Real Stage 1 predictions will have visual mistakes. If Sarvam-1 only sees pristine phonemes during training, it will fail on real inference.
   Inject 10–15% noise into the phonemes:
   * **Viseme Swaps (10% chance):**
     - `P` $\leftrightarrow$ `B` $\leftrightarrow$ `M`
     - `TD` $\leftrightarrow$ `TH` $\leftrightarrow$ `D`
     - `K` $\leftrightarrow$ `G`
     - `CH` $\leftrightarrow$ `J`
   * **Random Drops (3% chance):** Omit a phoneme.
   * **Random Duplications (3% chance):** Repeat an adjacent phoneme.

3. **Format as JSONL:**
   ```json
   {"prompt": "<S2S>\n<PHONEMES>\nN M S TD E | BH AA R TD\n</PHONEMES>\n<TEXT>\n", "target": "नमस्ते भारत"}
   ```

### 4. Recommended LoRA Training Hyperparameters
* **Base Model**: `sarvamai/sarvam-1` (2B parameters)
* **LoRA Rank ($r$)**: `16`
* **LoRA Alpha ($\alpha$)**: `32`
* **Dropout**: `0.05`
* **Target Modules**: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
* **Learning Rate**: `2e-4` with cosine decay
* **Epochs**: 3 to 5 epochs
* **Hardware**: Single 16GB+ VRAM GPU (T4 / A10G / RTX 3090 / Colab A100)

*(You can use the training template function `train_sarvam_lora()` located in [`Models/Sarvam.py`](file:///c:/Users/aayus/OneDrive/Desktop/VALLR/Models/Sarvam.py)).*

### 5. Handover Package for Teammate 2
Save the fine-tuned adapter into the repository under:
`checkpoints/sarvam_lora/`

The folder must contain:
1. `adapter_config.json`
2. `adapter_model.safetensors`
3. `tokenizer_config.json`
4. `special_tokens_map.json`

**Pre-Handover Verification Test:**
Run this Python snippet inside the repo to confirm compatibility:
```python
from Models.Sarvam import validate_sarvam_adapter_compatibility

report = validate_sarvam_adapter_compatibility("checkpoints/sarvam_lora")
print("Adapter Compatible:", report["is_compatible"])
assert report["is_compatible"] is True
```

---

## 🤝 End-to-End Joint Test (When Both Parts Are Ready)

Once Teammate 1 provides the preprocessed clips and Teammate 2 provides the LoRA checkpoint:

1. **Train Stage 1 on the dataset:**
   ```bash
   python main.py --mode train --data_dir ./hindi_dataset --save_model_path ./checkpoints/punarvaani_stage1.pth --epochs 25
   ```
2. **Run End-to-End Inference:**
   ```bash
   python main.py --mode infer --infer_video_path ./hindi_dataset/test_sample.mp4 --save_model_path ./checkpoints/punarvaani_stage1.pth --sarvam_adapter_path ./checkpoints/sarvam_lora
   ```
3. **Evaluate CER/WER:**
   ```bash
   python main.py --mode eval --data_dir ./hindi_dataset --save_model_path ./checkpoints/punarvaani_stage1.pth --sarvam_adapter_path ./checkpoints/sarvam_lora
   ```


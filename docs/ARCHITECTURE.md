# Punarvaani — Current Architecture (as built)

> **Audience:** the data owner, the Sarvam-1 LoRA owner, the training owner, and anyone integrating
> with them.
>
> **Status:** describes the code that exists today (November 2026). For the forward-looking research
> plan, see [`ARCHITECTURE_PLAN.md`](ARCHITECTURE_PLAN.md). If the two disagree, **this document
> wins** for the current behaviour.
>
> **One sentence:** silent video of a face → mouth crops → Stage 1 predicts Hindi phonemes with CTC
> → Stage 2 (Sarvam-1 + LoRA) turns the phoneme stream into Devanagari text.

---

## 1. System overview

```text
silent video
    │
    ▼
┌──────────────────────────────┐
│ Preprocessing                │
│ MediaPipe face + mouth crop  │
│ 224 × 224 RGB, all frames    │
└──────────────┬───────────────┘
               │  (T, 224, 224, 3)
               ▼
┌──────────────────────────────┐
│ STAGE 1 — ML_VALLR           │
│ Spatiotemporal ViT + CTC     │
│ output: 55-class phonemes    │
└──────────────┬───────────────┘
               │  "N A M A S TD E | BH AA R A TD"
               ▼
┌──────────────────────────────┐
│ STAGE 2 — Sarvam-1 + LoRA    │
│ phoneme → Devanagari text    │
└──────────────┬───────────────┘
               │
               ▼
        "नमस्ते भारत"
```

Why two stages:

- On the lips, `प` / `ब` / `म` are the same viseme. Visual-only recognition cannot reliably choose
  between them.
- Stage 1 produces a compact, interpretable phoneme sequence.
- Stage 2 uses Hindi grammar and vocabulary to resolve the look-alikes.

What each stage can and cannot fix:

- Stage 1 fixes “what mouth shapes were made”.
- Stage 2 fixes “which Hindi word was intended” when the mouth shapes are ambiguous.
- **Stage 2 cannot repair a Stage-1 model that reads lips badly.** If PER is very high, the LLM is
  guessing.

---

## 2. The two hard contracts

Everything the team produces must satisfy one of these two contracts.

| Producer | Artifact | Consumer | Contract |
|---|---|---|---|
| Data owner (Teammate 1) | `hindi_dataset/` (or an explicit `--data_dir`) | `Data/dataset.py` | `<speaker>/<clip>.mp4` + `<clip>.txt`, pure Devanagari, speaker folders, 1.5–3 s clips |
| LoRA owner (Teammate 2) | `checkpoints/sarvam_lora/` (or `--sarvam_adapter_path`) | `Models/Sarvam.py` | PEFT adapter for `sarvamai/sarvam-1`, exact prompt tags, tokenizer files |
| Training owner (Teammate 3) | `checkpoints/punarvaani_stage1.pth` | `main.py`, `inference.py` | Stage-1 `state_dict`, same architecture flags used at training |

If either contract is broken, the pipeline either raises a clear error or falls back to MOCK mode.
It never silently trains on the wrong input (MediaPipe missing is now a hard error when mouth
cropping is requested).

---

## 3. Stage 1 — data path in detail

### 3.1 On-disk dataset contract

```text
<data_dir>/
├── spk01/
│   ├── spk01_001.mp4
│   ├── spk01_001.txt
│   ├── spk01_002.mp4
│   └── spk01_002.txt
├── spk02/
│   └── ...
└── spk03/
    └── ...
```

Rules enforced by `Data/dataset.py`:

1. The **first path component** under `data_dir` is the speaker ID.
2. Every video must have a `.txt` with the **same base name**.
3. A video without a `.txt` is skipped; a `.txt` without a video is reported by preflight.
4. At least **two speaker folders** are required to make a speaker-disjoint split.
5. Files directly in the root are rejected unless `--allow_unknown_speakers` is passed (not
   recommended for real training).
6. Supported video extensions: `.mp4`, `.avi`, `.mov`, `.mkv`.

Clip rules:

- 1.5–3 s, frontal, one speaker, mouth unobstructed.
- 25 or 30 fps. The model reads all frames up to `--max_frames`.
- Transcripts: UTF-8 Devanagari only. Numbers spelled out. English loanwords transliterated.
  No punctuation except spaces and `।`.

### 3.2 Raw clips vs mouth-cropped clips

Two supported modes:

| Mode | How to train | Needs MediaPipe? |
|---|---|---|
| Raw clips, crop on the fly | `--crop_mode mouth` (default) | Yes |
| Already mouth-cropped | `--preprocessed` (equivalent to `--crop_mode none`) | No |

`scripts/preprocess_lips.py`:

- Reads every video under `--input_dir`.
- Detects the face, roll-corrects it, tracks the outer lips.
- Crops a square mouth ROI with margin, smooths the box across frames (EMA).
- Resizes to `--frame_size` (default 224).
- Writes the same relative `<speaker>/<clip>.mp4` tree under `--output_dir`.
- Copies the transcript next to the output.
- Writes `preprocessing_report.json` with successes and per-file failure reasons.

Mouth landmarks used: MediaPipe FaceMesh outer-lip indices
`61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317`.

If MediaPipe is missing and `crop_mode` is `mouth`/`face`, the dataset raises `RuntimeError`.
It no longer silently falls back to full-frame training.

### 3.3 Frame sampling and CTC time steps

`HindiVideoDataset` returns variable-length clips.

| Setting | Meaning |
|---|---|
| `--num_frames 0` | default: read **all** frames, strided, up to `--max_frames` |
| `--max_frames 96` | cap per clip (96 frames ≈ 3.8 s at 25 fps, 3.2 s at 30 fps) |
| `--frame_stride 1` | read every Nth frame |
| `--num_frames 16` | legacy mode: evenly sample exactly 16 frames |

The CTC time-step formula is:

```text
T_ctc = ceil(frames / tubelet_size) × temporal_upsample_factor
```

- `tubelet_size = 2`
- `temporal_upsample_factor = 4` by default
- So a 75-frame clip → `ceil(75/2) × 4 = 152` CTC steps.

The model pads the temporal axis to a multiple of 2, so odd frame counts are not dropped.

**CTC requires `T_ctc >= U + R`**, where:

- `U` = target phoneme count,
- `R` = number of adjacent repeated phonemes.

If `T_ctc < U + R`, the sample is invalid:

- `custom_collate` drops it and increments `dropped`.
- Dataset initialisation reports the invalid percentage.
- If it exceeds `--max_invalid_sample_pct` (default 10%), training warns.
- With `--strict_ctc`, it raises instead.

This is why very long sentences or clipped transcripts fail. Shorten the sentence or raise
`--max_frames`.

### 3.4 Batch contract

`Data/dataset.py` defines:

```python
@dataclass
class VSRItem:
    video: Tensor          # (T, 3, H, W), float32 in [0, 1]
    phoneme_ids: Tensor    # (U,), long
    text: str              # original Devanagari transcript
    speaker_id: str
    num_frames: int
```

`custom_collate` produces:

```python
@dataclass
class VSRBatch:
    videos: Tensor          # (B, T_max, 3, H, W), zero-padded
    labels: list[Tensor]    # per-sample phoneme IDs
    texts: list[str]
    video_lengths: Tensor   # (B,) real frame counts
    input_lengths: Tensor   # (B,) CTC steps per sample
    valid_mask: Tensor      # (B,) bool
    dropped: int
```

The model receives `video_lengths`, builds an attention padding mask, zeros padded frames before
the 3D stem, and returns padded logits. CTC uses per-sample `input_lengths`.

### 3.5 Hindi phoneme inventory (55 classes)

Defined in `config.py`, IDs 0–54:

| ID | Token | Meaning |
|---:|---|---|
| 0 | `<pad>` | CTC blank |
| 1 | `\|` | word boundary |
| 2–14 | `A AA I II U UU E AI O AU RI AN AH` | vowels / nasal / visarga |
| 15–19 | `K KH G GH NG` | velar |
| 20–24 | `CH CHH J JH NY` | palatal |
| 25–31 | `TR THR DR DHR NR RD RHD` | retroflex + flaps |
| 32–36 | `TD THD DD DHD N` | dental |
| 37–41 | `P PH B BH M` | labial |
| 42–45 | `Y R L V` | approximants |
| 46–49 | `SH SHR S H` | sibilants / fricatives |
| 50–54 | `Q KH_LOAN GH_LOAN Z F` | loan consonants |

The G2P engine (`Data/hindi_g2p.py`) maps Devanagari to these tokens and applies Hindi schwa
deletion. Examples that are pinned by tests:

```text
नमस्ते  -> N A M A S TD E
भारत    -> BH AA R A TD
कमल     -> K A M A L
कमला     -> K A M L AA
फ़िल्म   -> F I L M
समझना   -> S A M A JH N AA
डॉक्टर  -> DR O K TR A R
```

Important caveats for the data owner:

- The G2P is rule-based and approximate for loanwords and compounds.
- Every wrong transcript becomes label noise; the model will learn it.
- English letters, digits, and unsupported characters are logged in the G2P diagnostic report and
  surfaced by `scripts/preflight.py`.
- The word boundary `|` is part of the target sequence.

### 3.6 Speaker-disjoint splitting

`HindiVideoDataset` supports three split modes:

1. **Single `--data_dir` with `--val_split`** (e.g. `0.2`):
   - Splits **speakers**, not clips.
   - `n_val = max(1, round(n_speakers × val_split))`, capped so training is never empty.
   - The same speaker never appears in both train and val.
2. **Dedicated `--val_dir`** (recommended for the final run):
   - Train uses `--data_dir`; validation uses `--val_dir` as a separate speaker set.
   - Pass `--val_split 0` for the training dataset so it uses all its speakers.
3. **`--allow_unknown_speakers`**:
   - Only for quick debugging when clips sit directly in the root.
   - Falls back to a clip-level split, which can leak speakers. Do not use for real training.

For a held-out test set, use a third folder (`data/test/`) and only touch it for final evaluation.

### 3.7 Stage-1 model in detail

`Models/ML_VALLR.py`:

```text
video (B, 3, T, 224, 224)
  │  zero padded frames before the stem
  ▼
Conv3d tubelet: kernel 2×16×16, stride 2×16×16
  │  (B, D, Tp, 14, 14), Tp = ceil(T/2)
  ▼
optional spatial token pooling (default 7×7)
  │  (B, D, Tp, 7, 7)
  ▼
flatten + positional embedding + ViT blocks
  │  (B, N, D), N = Tp × 49
  ▼
SpatialAttentionPool over the 49 spatial tokens
  │  (B, Tp, D)
  ▼
TemporalAdapter: Conv1d → ConvTranspose1d (×4) → Conv1d
  │  (B, T_ctc, adapter_dim)
  ▼
Linear + ReLU → CTC head
  │
  ▼
logits (B, T_ctc, 55), features (B, T_ctc, 256)
```

Defaults:

| Parameter | Value |
|---|---|
| `feature_size` (ViT width) | 768 |
| `vit_depth` | 8 |
| `vit_heads` | 12 |
| `patch_size` | 16 |
| `tubelet_size` | 2 |
| `img_size` | 224 |
| `adapter_dim` | 256 |
| `ctc_hidden_size` | 256 |
| `temporal_upsample_factor` | 4 |
| `spatial_pool_tokens` | 7×7 (CLI default) |

Padding correctness:

- Padded frames are zeroed **before** the 3D stem.
- Padded tokens are excluded from attention with a key-padding mask.
- Padded tubelets are zeroed before the temporal adapter.
- Padded CTC steps are zeroed after the adapter.
- Tests assert that a padded sample produces the same logits as the same sample run alone.

### 3.8 Stage-1 training loop

`main.py --mode train`:

- AdamW, `lr=1e-4`, `weight_decay=1e-4`.
- Linear warmup (`--warmup_steps`, default 2000) then `ReduceLROnPlateau` (factor 0.5, patience 3).
- Gradient clipping at 0.5.
- Mixed precision:
  - CUDA: fp16 autocast + `GradScaler`.
  - CPU: bf16 autocast, no scaler (smoke tests only).
  - CTC log-softmax/loss is forced to **fp32**.
- Saves the model whenever **validation PER** improves.
- Prints one line per epoch:

```text
Epoch [3/25] | Train Loss: 3.8120 (PER: 78.4%, dropped 0) | Val Loss: 4.1050 (PER: 84.1%, dropped 0) | LR: 1.00e-04
```

What is saved:

- Only the model `state_dict` at `--save_model_path`.
- **Not** saved: optimizer state, scheduler state, epoch number, RNG state.
- There is no resume-from-checkpoint. A crashed run restarts from random initialization.
- The file is the **best validation-PER** checkpoint, not necessarily the final epoch.

### 3.9 Stage-1 metrics

- **CTC loss** — average negative log-likelihood.
- **PER** — phoneme error rate after greedy CTC decoding (substitutions + insertions + deletions
  divided by reference phonemes). Can exceed 100% early.
- **dropped** — samples excluded by the `T_ctc >= U + R` rule.
- Evaluation also prints CTC loss, PER, and dropped count.

PER is not WER/CER. Stage 2 CER/WER is the product-level metric, and only meaningful when Stage 1
is already reasonable.

---

## 4. Stage 2 — Sarvam-1 LoRA connector in detail

### 4.1 Role

Stage 2 receives the Stage-1 phoneme string and returns Devanagari text. It resolves viseme
ambiguity using Hindi grammar and vocabulary.

Example:

```text
input : N A M A S TD E | BH AA R A TD
output: नमस्ते भारत
```

### 4.2 Exact prompt contract

`Models/Sarvam.py` defines:

```python
TAGS = ["<S2S>", "<PHONEMES>", "</PHONEMES>", "<TEXT>"]
```

The prompt is exactly:

```text
<S2S>
<PHONEMES>
{phoneme_sequence}
</PHONEMES>
<TEXT>
```

The model must answer with the Devanagari sentence only.

Training rows are JSONL with two keys:

```json
{"prompt": "<S2S>\n<PHONEMES>\nN A M A S TD E | BH AA R A TD\n</PHONEMES>\n<TEXT>\n", "target": "नमस्ते भारत"}
```

### 4.3 Tokenizer and embedding rules

Before training:

```python
tokenizer.add_special_tokens({"additional_special_tokens": TAGS})
model.resize_token_embeddings(len(tokenizer))
```

Each tag must encode as **one token**:

```python
assert len(tokenizer.encode("<S2S>", add_special_tokens=False)) == 1
```

If the tokenizer splits a tag into subwords, the adapter must not be mounted. Preflight reports
this as a blocking issue.

### 4.4 Adapter artifact contract

```text
checkpoints/sarvam_lora/
├── adapter_config.json
├── adapter_model.safetensors      (or adapter_model.bin)
├── tokenizer_config.json
├── special_tokens_map.json
└── tokenizer.json                 (if exported)
```

`adapter_config.json` must declare:

```json
{"base_model_name_or_path": "sarvamai/sarvam-1"}
```

The loader uses the adapter's tokenizer if `tokenizer_config.json` is present; otherwise it uses
the base model tokenizer and adds the four tags.

### 4.5 Loading, validation and MOCK mode

`HindiPhonemeTranslator` does:

1. Choose tokenizer source: adapter directory if it has `tokenizer_config.json`, else base model.
2. Load tokenizer.
3. Add any missing special tokens.
4. Set `pad_token` to `eos_token` if absent.
5. Load base model in fp16 on CUDA (or fp32 on CPU).
6. Resize embeddings.
7. Validate the adapter:
   - directory exists,
   - `adapter_config.json` exists,
   - base model matches `sarvamai/sarvam-1`,
   - tokenizer contains all four tags and each is one token.
8. Mount with `PeftModel.from_pretrained`.

If any step fails, it **falls back to MOCK mode** and records `mock_reason`. It never silently
mounts a mismatched adapter.

MOCK mode:

- `translate()` returns `[MOCK TRANSCRIPTION: <phonemes>]`.
- Evaluation prints the reason and **skips CER/WER**.
- The rest of the pipeline still runs, so Stage 1 can be tested without the adapter.

### 4.6 Stage-2 training data

The adapter is trained on **text only**, not video:

```text
Hindi corpus
  → clean Devanagari (drop English/digits/punctuation)
  → Data.hindi_g2p.devanagari_to_phonemes
  → inject synthetic visual-speech noise
  → JSONL {prompt, target}
```

Recommended synthetic noise:

- 10% viseme swaps inside groups such as `P/B/M`, `TD/THD/DD/DHD/N`, `TR/THR/DR/DHR/NR`,
  `K/KH/G/GH/NG`, `CH/CHH/J/JH/NY`, `PH/BH`.
- 3% random phoneme drops.
- 3% adjacent duplications.

Why noise matters: real Stage-1 output will contain mistakes. Training only on perfect phonemes
makes the adapter brittle at inference.

### 4.7 Training function and hyperparameters

`Models/Sarvam.py::train_sarvam_lora`:

```python
train_sarvam_lora(
    train_jsonl_path="sarvam_data/train.jsonl",
    val_jsonl_path="sarvam_data/val.jsonl",
    output_dir="checkpoints/sarvam_lora",
    base_model_id="sarvamai/sarvam-1",
    epochs=5,
    batch_size=4,
    lr=2e-4,
)
```

- Base: `sarvamai/sarvam-1` (2B).
- LoRA r=16, alpha=32, dropout=0.05.
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- fp16 training, GPU with at least 16 GB VRAM.
- Prompt tokens are masked out of the loss; only the target text is learned.
- Saves the adapter and tokenizer to `output_dir`.

### 4.8 What breaks Stage 2

| Problem | Result |
|---|---|
| Adapter trained on a different base model | Validation fails, MOCK mode |
| Missing `adapter_model.safetensors` or `.bin` | Validation fails, MOCK mode |
| Missing tags in tokenizer | Validation fails, MOCK mode |
| Tags split into subwords | Validation fails, MOCK mode |
| `peft`/`transformers` not installed | MOCK mode with dependency reason |
| Adapter trained without noise | Poor correction on real Stage-1 output |
| Stage 1 PER very high | Stage 2 produces fluent but wrong text |

---

## 5. Inference path

`inference.py::transcribe_video` returns a dictionary:

```python
{
    "success": bool,
    "phonemes": str,
    "transcription": str | None,
    "stage1_status": "ok" | "model_missing" | "failed",
    "stage2_status": "ok" | "adapter_missing" | "mock_mode" | "failed",
    "error": str | None,
    "num_frames": int,
}
```

Flow:

1. Load frames (`--crop_mode mouth` requires MediaPipe; `none` for pre-cropped video).
2. Build `ML_VALLR` with the same architecture flags used for training.
3. Forward with `video_lengths`.
4. Greedy CTC decode → phoneme string.
5. `HindiPhonemeTranslator` → Devanagari (or MOCK).

`main.py --mode infer` prints the predicted phonemes and final Hindi text.

Important: the Stage-1 architecture flags (`--vit_embed_dim`, `--vit_depth`, `--vit_heads`,
`--temporal_upsample_factor`, `--spatial_pool_tokens`) must match the checkpoint. A mismatch
produces a load error, not silently wrong predictions.

---

## 6. What each teammate must produce

### 6.1 Data owner (Teammate 1)

Required:

- `<data_dir>/<speaker>/<clip>.mp4` + `<clip>.txt`.
- Speaker-disjoint `train` / `val` / `test` folders (or a single folder with `--val_split`).
- Pure Devanagari transcripts, numbers spelled out.
- 1.5–3 s clips, frontal, mouth visible.
- `preprocessing_report.json` with ≥95% crop success (if you run the cropper).
- `preflight_report.json` with `Preflight OK` for the training split (use `--min_speakers 1` for val/test).
- At least 2 speakers for training; at least 1 for val/test (`--min_speakers 1` in preflight).

What breaks downstream:

| Mistake | Downstream effect |
|---|---|
| Loose files, no speaker folder | `ValueError` at dataset load |
| Missing `.txt` | Video skipped; preflight fails |
| English/digits in transcript | G2P label noise; preflight warning |
| Clips too long for `T_ctc` | Sample dropped; `dropped` increases |
| Same speaker in train and val | Split is rejected / evaluation is meaningless |
| Whole-face crops instead of mouth | Model learns the wrong input; training looks fine but fails |
| Data in OneDrive | Slow/locked reads; training stalls |

### 6.2 LoRA owner (Teammate 2)

Required:

- Adapter trained from exact base `sarvamai/sarvam-1`.
- `TAGS` registered before embedding resize.
- Each tag encodes as one token.
- Adapter files and tokenizer files in one directory.
- Training data built from a Hindi corpus with G2P + synthetic noise.
- Validation via `validate_sarvam_adapter_compatibility` and `scripts/preflight.py`.
- A short note: rows, epochs, loss, corpus source.

What breaks downstream:

| Mistake | Downstream effect |
|---|---|
| Wrong base model | MOCK mode; CER/WER skipped |
| Only `adapter_config.json` copied | MOCK mode; missing weights |
| Tags split into subwords | MOCK mode; validation fails |
| Clean-only phoneme training | Poor real-world correction |
| Missing tokenizer files | Base tokenizer used; may still work, but verify tags |
| Adapter trained on another prompt format | Garbage text |

### 6.3 Training owner (Teammate 3)

- Runs `scripts/preflight.py` on data + adapter.
- Runs the four gates in `docs/TRAINING_RUNBOOK.md`.
- Saves the best Stage-1 checkpoint.
- Evaluates on held-out test speakers.
- Hands back the checkpoint, command, logs, and metrics.

### 6.4 QA owner

- Unit tests, preflight, adapter validation, end-to-end inference.
- Confirms no speaker leakage and no data/weights in git.

---

## 7. File and artifact map

| Path | What it is | Git? |
|---|---|---|
| `Data/dataset.py` | dataset, splits, collate, CTC checks | tracked |
| `Data/hindi_g2p.py` | Devanagari → phonemes | tracked |
| `Data/metrics.py` | CER/WER helpers | tracked |
| `Data/synthetic.py` | synthetic smoke dataset | tracked |
| `Models/ML_VALLR.py` | Stage 1 | tracked |
| `Models/Sarvam.py` | Stage 2 connector + adapter validation | tracked |
| `face_cropper.py` | MediaPipe face/mouth crop | tracked |
| `mouth_geometry.py` | mouth box maths | tracked |
| `scripts/preprocess_lips.py` | offline mouth cropping | tracked |
| `scripts/preflight.py` | dataset/adapter/runtime checks | tracked |
| `scripts/make_synthetic_dataset.py` | smoke dataset | tracked |
| `main.py` | train / eval / infer | tracked |
| `config.py` | CLI and 55-class vocabulary | tracked |
| `inference.py` | single-video API | tracked |
| `tests/` | unit/integration tests | tracked |
| `hindi_dataset/`, `data/` | dataset | ignored |
| `checkpoints/` | Stage-1 weights and LoRA adapter | ignored |
| `logs/`, `reports/`, `outputs/`, `runs/` | run artifacts | ignored |

---

## 8. Configuration reference (most-used flags)

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `train` | `train`, `eval`, `infer` |
| `--data_dir` | None | root with `<speaker>/<clip>.mp4 + .txt` |
| `--val_dir` | None | separate validation/test root |
| `--val_split` | 0.1 | fraction of **speakers** held out when no `--val_dir` |
| `--num_frames` | 0 | 0 = all frames up to `--max_frames`; >0 = legacy uniform sample |
| `--max_frames` | 96 | cap per clip |
| `--frame_stride` | 1 | read every Nth frame |
| `--crop_mode` | `mouth` | `mouth`, `face`, `none` |
| `--preprocessed` | False | force `crop_mode none` |
| `--batch_size` | 4 | training batch |
| `--eval_batch_size` | 4 | validation/eval batch |
| `--epochs` | 20 | training epochs |
| `--warmup_steps` | 2000 | LR warmup |
| `--max_invalid_sample_pct` | 10.0 | allowed dropped-sample percentage |
| `--strict_ctc` | False | raise instead of warn when the limit is exceeded |
| `--temporal_upsample_factor` | 4 | CTC upsampling factor |
| `--vit_embed_dim` | 768 | ViT width |
| `--vit_depth` | 8 | ViT blocks |
| `--vit_heads` | 12 | attention heads |
| `--spatial_pool_tokens` | `7x7` | spatial token pooling or `none` |
| `--save_model_path` | `./checkpoints/ml_vallr_hindi.pth` | Stage-1 checkpoint |
| `--sarvam_adapter_path` | None | LoRA adapter directory |
| `--sarvam_base_model` | `sarvamai/sarvam-1` | Stage-2 base model |
| `--device` | `cuda` | `cuda` or `cpu` |

---

## 9. Current limitations vs the plan

Implemented today:

- Hindi 55-class phoneme inventory + G2P.
- All-frames, variable-length input with masks.
- Speaker-disjoint splits.
- Mouth ROI cropping.
- Stage-1 CTC training/eval/inference.
- Stage-2 Sarvam-1 LoRA loading, validation, MOCK fallback, training template.
- Preflight for dataset, adapter, runtime.
- 73 automated tests (including slow E2E).

Not implemented yet (see `ARCHITECTURE_PLAN.md`):

- Devanagari character CTC head and viseme CTC head.
- n-best phoneme/character input to Sarvam-1.
- Per-patient adaptation (padding/LoRA).
- Phrase mode + confidence rejection.
- Data engine (IndicWhisper/WhisperX, SyncNet).
- Voice/TTS output.
- Stage-1 beam search (greedy decode only).
- Resume-from-checkpoint.
- VER metric (viseme error rate).
- W&B/TensorBoard logging.

---

## 10. Verification commands

```bash
# Unit/integration tests
python -m unittest discover -s tests -v

# Dataset preflight (crop on the fly)
python scripts/preflight.py --data_dir ./hindi_dataset --crop_mode mouth

# Dataset preflight (already cropped)
python scripts/preflight.py --data_dir ./hindi_dataset --crop_mode none

# Adapter validation
python -c "from Models.Sarvam import validate_sarvam_adapter_compatibility as v; print(v('checkpoints/sarvam_lora'))"

# Adapter + dataset together
python scripts/preflight.py --data_dir ./hindi_dataset --sarvam_adapter_path ./checkpoints/sarvam_lora
```

---

## 11. Glossary

| Term | Meaning |
|---|---|
| **Viseme** | A group of phonemes that look identical on the lips (`P/B/M`). |
| **Phoneme** | A basic speech sound (`M`, `AA`, `TD`). |
| **G2P** | Grapheme-to-phoneme conversion. |
| **HPI** | Hindi Phonetic Inventory; the 55-class target set. |
| **CTC** | Connectionist Temporal Classification; aligns frames to labels without frame-level labels. |
| **PER** | Phoneme Error Rate. |
| **CER / WER** | Character / Word Error Rate. |
| **LoRA** | Low-rank adapter for a large model. |
| **MOCK mode** | Stage 2 disabled because the adapter is missing or invalid. |
| **T_ctc** | Number of CTC time steps produced by the temporal adapter. |
| **U + R** | Minimum CTC length: target length plus adjacent repeats. |

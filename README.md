# Punarvaani (पुनर्वाणी): Visual Speech Recognition for Hindi

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch: 2.4+](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![Base Architecture: VALLR (ICCV 2025)](https://img.shields.io/badge/Based%20On-VALLR%20(ICCV%202025)-red.svg)](https://arxiv.org/abs/2503.21408)

> **Punarvaani** (*Punar* = Re-/Restore, *Vaani* = Voice/Speech) is an end-to-end, two-stage **Visual Automatic Speech Recognition (VASR) / Lip-Reading system tailored for the Hindi language**. By reading silent mouth movements from video, Punarvaani predicts Hindi phoneme sequences and uses an Indic Large Language Model to restore complete, grammatically fluent Devanagari Hindi speech.

> **New to the codebase?**
> - [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — as-built pipeline, tensor shapes, and the exact dataset/LoRA contracts.
> - [`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md) — clone-to-checkpoint training guide with four pre-training gates.
> - [`TEAM_GUIDE.md`](TEAM_GUIDE.md) — beginner onboarding for the data and LoRA roles.

---

## 🙏 Acknowledgments & Foundational Architecture

**Punarvaani** is built upon the pioneering research and architectural framework of **VALLR** (**V**isual **A**SR **L**anguage Model for **L**ip **R**eading), published at **ICCV 2025**:

> **[ICCV 2025] VALLR: Visual ASR Language Model for Lip Reading**  
> *Marshall Thomas, Edward Fish, Richard Bowden (University of Surrey)*  
> 📄 **Paper:** [arXiv:2503.21408](https://arxiv.org/abs/2503.21408)  
> 🔗 **Original Repository:** [MarshallT-99/VALLR](https://github.com/MarshallT-99/VALLR)

### Our Respect & Gratitude to the Creators of VALLR
We extend our deepest gratitude to Marshall Thomas, Edward Fish, and Prof. Richard Bowden for their breakthrough research. Prior to VALLR, lip-reading models required thousands of hours of self-supervised video pre-training and struggled heavily with *viseme ambiguity* (different sounds producing identical mouth shapes). 

The VALLR authors revolutionized the field by demonstrating that **decoupling visual phoneme recognition from linguistic sentence reconstruction** achieves state-of-the-art accuracy with **99.4% less labeled training data**. 

**Punarvaani** honors and extends this foundational work by bringing this state-of-the-art paradigm into the **Indic linguistic ecosystem**, introducing:
1. A **55-class Hindi Phonetic Inventory (HPI)** and rule-based Hindi Grapheme-to-Phoneme engine with Schwa Deletion.
2. An adapted **Spatiotemporal ViT CTC model** trained on Hindi visual speech.
3. An Indic Large Language Model (**Sarvam-1**) fine-tuned via LoRA for phoneme-to-Devanagari text reconstruction.

---

## 🌟 The End Product: What Happens After Dataset, Training & LLM Fine-Tuning?

When the dataset is collected, the Stage 1 model is trained, and the Stage 2 LLM is fine-tuned, **Punarvaani becomes a complete "silent speech-to-text" engine**. 

```
┌─────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│   SILENT VIDEO  │      │  PREPROCESSING & CROP   │      │ STAGE 1: VISION-TO-CTC  │      │ STAGE 2: SARVAM-1 LLM   │
│                 │ ───► │                         │ ───► │                         │ ───► │                         │
│ Speaker talking │      │ MediaPipe Face Mesh     │      │ Spatiotemporal ViT      │      │ Indic Language Model    │
│ without audio   │      │ Tracks mouth ROI        │      │ Outputs Hindi Phonemes  │      │ Reconstructs Sentences  │
└─────────────────┘      └─────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
                                                                       │                                │
                                                                       ▼                                ▼
                                                            Predicted Phonemes:                 Final Transcription:
                                                            "N A M A S TD E | BH AA R A TD"      "नमस्ते भारत"
```

### The Step-by-Step Inference Journey

1. **Input: Silent Video Feed**  
   The user supplies a video of a person speaking Hindi without any audio track (e.g., silent CCTV, high-noise factory/cockpit footage, silent speech from aphonic individuals, or a muted video call).

2. **Step 1: Intelligent Mouth Tracking & ROI Isolation (`face_cropper.py`)**  
   MediaPipe Face Mesh detects facial landmarks across the video clip. It tracks the lips regardless of minor head rotations, crops the mouth region of interest (ROI) to $224 \times 224$ pixels, and reads **all frames** of the clip (optionally strided and capped by `--max_frames`) so long sentences are not truncated.

3. **Step 2: Spatiotemporal Visual Recognition (`ML_VALLR`)**  
   The cropped mouth frames are passed through the 3D tubelet-convolutional Vision Transformer (ViT). The model evaluates subtle kinematic muscle movements (opening, rounded lips, tongue presses, dental contact) and emits frame-wise logits over the 55 Hindi phonetic classes. Greedy/beam CTC decoding resolves these into a clean phoneme string:
   $$\text{Video Frames} \xrightarrow{\text{ViT + CTC}} \texttt{N A M A S TD E | BH AA R A TD}$$

4. **Step 3: Linguistic Resolution via Fine-Tuned LLM (`Models/Sarvam.py`)**  
   In visual speech, many sounds look identical on the lips—for example, **/P/**, **/B/**, and **/M/** are all bilabial closures and visual twins (visemes). Visuals alone cannot distinguish "बात" (*baat*) from "पात" (*paat*).  
   
   This is where the fine-tuned **Sarvam-1 Indic LLM** acts as the system's linguistic brain:
   * It takes the noisy phoneme stream as a prompt:
     ```text
     <S2S>
     <PHONEMES>
     N A M A S TD E | BH AA R A TD
     </PHONEMES>
     <TEXT>
     ```
   * Leveraging its deep prior knowledge of Hindi syntax, vocabulary, and grammar, it corrects phonetic ambiguity and restores proper spelling and matras.

5. **Output: Coherent Devanagari Hindi Text**  
   The end product instantly returns the decoded Hindi transcript:
   $$\mathbf{नमस्ते\ भारत}$$

---

## 🏛️ System Architecture

| Component | Architecture / Technology | Role in Punarvaani |
| :--- | :--- | :--- |
| **Face & Lip Cropper** | MediaPipe Face Landmark Detection (`scripts/preprocess_lips.py`) | Stabilizes, normalizes, and crops speaker mouth frames ($224 \times 224$) with EMA smoothing |
| **G2P Engine** | Pure-Python Devanagari Rule-based G2P (`Data/hindi_g2p.py`) | Handles 55-class Hindi Phonetic Inventory (HPI), Schwa Deletion, independent vowels, and audit logs |
| **Stage 1: Vision Model** | Spatiotemporal ViT + Spatial Attention Pooling + Temporal Adapter (`Models/ML_VALLR.py`) | Extracts tubelet tokens, pools spatial patches ($196 \to 1$), performs learned temporal upsampling, and aligns phonemes via CTC ($T \ge U + R$) |
| **Stage 2: Language Model** | `sarvamai/sarvam-1` (2B Parameters + LoRA) (`Models/Sarvam.py`) | Resolves viseme ambiguities and reconstructs grammatically fluent Devanagari text |

---

## 📁 Repository Structure

```text
Punarvaani/
├── config.py                 # Configuration arguments, hyperparameters & 55-class Hindi vocab
├── main.py                   # Unified CLI for training, evaluation, and end-to-end inference
├── inference.py              # Standalone Python inference API for web/downstream integration
├── face_cropper.py           # Real-time MediaPipe face and mouth region detector
├── requirements.txt          # Python dependencies
│
├── Data/
│   ├── dataset.py            # Video dataset loader, collation with CTC T >= U + R, and speaker split check
│   └── hindi_g2p.py          # Hindi G2P converter with Schwa Deletion & diagnostic audit logging
│
├── Models/
│   ├── ML_VALLR.py           # Spatiotemporal ViT, SpatialAttentionPool, and TemporalAdapter
│   ├── Sarvam.py             # Sarvam-1 connector with 7-step adapter validation & MOCK fallback
│   └── VALLR.py              # Classic VALLR (VideoMAE + Wav2Vec2 CTC) reference implementation
│
├── mouth_geometry.py         # Dependency-free mouth ROI box maths (tested without MediaPipe)
│
├── scripts/
│   ├── preprocess_lips.py    # Offline batch MediaPipe mouth cropping (preserves speaker folders)
│   ├── preflight.py          # Validate a dataset handover + Sarvam adapter before training
│   └── make_synthetic_dataset.py  # Tiny synthetic speaker-tree dataset for smoke tests
│
├── tests/
│   ├── __init__.py
│   ├── test_punarvaani.py    # Core phonetics, ViT, CTC, collate, Sarvam contract
│   ├── test_dataset.py       # Speaker-disjoint splits + all-frames sampling
│   ├── test_collate.py       # Variable-length padding and CTC lengths
│   ├── test_model_lengths.py # Masking, padding invariance, input lengths
│   ├── test_mouth_geometry.py
│   ├── test_preprocess.py
│   ├── test_preflight.py
│   ├── test_sarvam.py / test_scheduler.py / test_mixed_precision.py / test_g2p.py
│   └── test_smoke_e2e.py     # RUN_SLOW=1 train -> eval -> infer
│
├── hindi_dataset/            # Directory for paired video files and Devanagari text transcripts
│   ├── README.md             # Dataset instructions
│   ├── sample001.mp4
│   └── sample001.txt
│
└── checkpoints/
    └── sarvam_lora/          # Target folder for fine-tuned Sarvam-1 LoRA weights
        └── README.md
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Installation

```bash
# Clone the repository
git clone https://github.com/MarshallT-99/VALLR.git Punarvaani
cd Punarvaani

# Install required dependencies
pip install -r requirements.txt
```

### 2. Preparing the Hindi Dataset & Offline Lip Cropping

Store paired raw video files and text transcripts in **speaker subfolders** (the split is speaker-disjoint):

```text
raw_videos/
├── spk01/
│   ├── spk01_001.mp4
│   └── spk01_001.txt      # UTF-8 Devanagari, e.g. "नमस्ते भारत"
└── spk02/...
```

Optionally run the offline batch preprocessor to crop and stabilize lips ahead of training:

```bash
python scripts/preprocess_lips.py \
  --input_dir ./raw_videos \
  --output_dir ./hindi_dataset \
  --max_frames 96 \
  --frame_size 224
```

### 3. Training Stage 1 (Video-to-Phoneme CTC Model)

Train the visual feature extractor and CTC head on your Hindi dataset:

```bash
python main.py \
  --mode train \
  --language hindi \
  --data_dir ./hindi_dataset \
  --save_model_path ./checkpoints/punarvaani_stage1.pth \
  --epochs 25 \
  --batch_size 8 \
  --max_frames 96 \
  --temporal_upsample_factor 4
```

### 4. Evaluating Model Performance

Evaluate validation CTC loss and Phoneme Error Rate (PER) on held-out test data:

```bash
python main.py \
  --mode eval \
  --data_dir ./hindi_dataset \
  --save_model_path ./checkpoints/punarvaani_stage1.pth \
  --eval_batch_size 8
```

### 5. Running End-to-End Inference via CLI

Once Stage 1 is trained and fine-tuned Sarvam-1 LoRA weights are in `checkpoints/sarvam_lora/`, run end-to-end inference on any silent video:

```bash
python main.py \
  --mode infer \
  --infer_video_path ./test_video.mp4 \
  --save_model_path ./checkpoints/punarvaani_stage1.pth \
  --sarvam_adapter_path ./checkpoints/sarvam_lora
```

*(Note: If Sarvam-1 LoRA weights are still training, omitting `--sarvam_adapter_path` runs Stage 2 in transparent MOCK mode so you can verify Stage 1 predictions).*

### 6. Standalone Python Inference API

For web interfaces (FastAPI / Streamlit / Flask), import and run `transcribe_video`:

```python
from inference import transcribe_video

result = transcribe_video(
    video_path="path/to/silent_video.mp4",
    stage1_model_path="checkpoints/punarvaani_stage1.pth",
    sarvam_adapter_path="checkpoints/sarvam_lora"
)

if result["success"]:
    print("Predicted Phonemes:", result["phonemes"])
    print("Hindi Transcription:", result["transcription"])
else:
    print("Transcription failed:", result["error"])
```

### 7. Running the Automated Test Suite

Run the unit suite covering phonetics, ViT space-time interpolation, variable-length padding, CTC $T \ge U + R$ conditions, speaker-disjoint splits, the preflight validator, and Sarvam token compatibility:

```bash
python -m unittest discover -s tests -v
```

Before training on a new handover, validate the data and adapter:

```bash
python scripts/preflight.py \
  --data_dir ./hindi_dataset \
  --sarvam_adapter_path ./checkpoints/sarvam_lora
```

---

## 📚 Citations & Academic References

If you use **Punarvaani** or build upon this work, please cite both the foundational **VALLR** paper and the **Sarvam-1** model:

### Foundational Paper: VALLR (ICCV 2025)
```bibtex
@article{thomas2025vallr,
  title={VALLR: Visual ASR Language Model for Lip Reading},
  author={Thomas, Marshall and Fish, Edward and Bowden, Richard},
  journal={arXiv preprint arXiv:2503.21408},
  year={2025}
}
```

### Indic Language Model: Sarvam-1
```bibtex
@article{sarvam2024sarvam1,
  title={Sarvam-1: An Efficient 2B Language Model Optimized for Indic Languages},
  author={Sarvam AI},
  year={2024},
  url={https://huggingface.co/sarvamai/sarvam-1}
}
```

---

## 📄 License
This repository is released under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**, preserving the academic and research-oriented spirit of the original VALLR implementation.

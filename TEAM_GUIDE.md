# Punarvaani — Teammate Handbook (start here)

> **Who this is for:** teammates who have never seen this codebase, may never have used a terminal, and need to produce one part of the project and hand it back.
>
> **How to use this guide:** read Sections 0–4 once, then jump to your role. Copy commands exactly, one block at a time. Do not type the `PS C:\...>` or `$` symbols you see in examples — those are just the prompt.
>
> If anything in this guide does not match what you see on your screen, stop and ask the team lead before changing files.
>
> **Training the Stage-1 model on the college Windows/A5000 PC?** Follow the dedicated
> [`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md) instead of Part 3 below. It covers the
> full setup, data placement, the four pre-training gates, full training, monitoring and evaluation.

---

## 0. What is this project, in plain language?

Punarvaani is a system that **reads a person's lips from a silent video and writes down what they said in Hindi**.

It works in two stages:

```text
silent video of a face
        │
        ▼
STAGE 1: look at the mouth movements
        │
        ▼
a sequence of Hindi sounds, called "phonemes"
        │
        ▼
STAGE 2: a Hindi language model fixes the ambiguous sounds
        │
        ▼
final Hindi text (Devanagari)
```

Two examples of why Stage 2 is needed:

- On the lips, `प` (pa), `ब` (ba), and `म` (ma) look almost identical.
- Visual-only reading might produce `भुझे बानी चाहिए`; the language model should correct it to `मुझे पानी चाहिए` ("I need water").

The team produces three things:

| What | Who | Final folder/file |
|---|---|---|
| A clean Hindi video dataset with transcripts | Teammate 1 (Data) | `hindi_dataset/` |
| A fine-tuned Sarvam-1 language model adapter | Teammate 2 (LLM) | `checkpoints/sarvam_lora/` |
| A trained Stage-1 lip-reading model | Teammate 3 (Training) | `checkpoints/punarvaani_stage1.pth` |

If your team has only two people, Teammates 1 and 2 are the core roles, and the lead handles training and quality checks.

---

## 1. Words you will hear (mini glossary)

| Word | Plain meaning |
|---|---|
| **Repository / repo** | The project folder containing all code. |
| **Terminal** | The text window where you type commands. |
| **Virtual environment (`venv`)** | A private box of Python packages for this project, so it does not break other projects. |
| **Dependency / package** | A library the code needs (PyTorch, OpenCV, MediaPipe, etc.). |
| **Phoneme** | A basic speech sound, e.g. `M`, `AA`, `TD`. |
| **G2P** | "Grapheme to Phoneme": converts Devanagari spelling into phonemes. |
| **Viseme** | A group of sounds that look the same on the lips (`P/B/M`). |
| **CTC** | The training method that lets the model align sounds to video frames without frame-by-frame labels. |
| **Epoch** | One full pass over the training data. |
| **Checkpoint** | A saved file of trained model weights (`.pth`). |
| **LoRA / adapter** | A small set of extra weights that adapt a large pretrained model. Much smaller than the full model. |
| **JSONL** | A text file where each line is one JSON object. Used for LLM training data. |
| **Inference** | Running a trained model on new input (a video) to get a prediction. |

---

## 2. First-time setup (everyone does this once)

### 2.1 Install the two tools you need

1. **Python 3.10 or 3.11** from <https://www.python.org/downloads/>
   - On Windows, tick **"Add python.exe to PATH"** during installation.
   - Do **not** use Python 3.13 for now: some video libraries do not have ready-made installers for it yet.
2. **Git** from <https://git-scm.com/downloads>
   - If you cannot install Git, ask the team lead for a ZIP of the project instead.

Optional but recommended: **Visual Studio Code** from <https://code.visualstudio.com/> to open `.py` and `.txt` files.

### 2.2 Open a terminal

- **Windows:** press the Windows key, type `PowerShell`, press Enter. A blue window appears.
- **macOS:** press `Command + Space`, type `Terminal`, press Enter.
- **Linux:** open your Terminal app.

Check Python:

```bash
python --version
```

Expected: `Python 3.10.x` or `Python 3.11.x`.

On Windows, if `python` is not found, try:

```bash
py -3.11 --version
```

If that works, use `py -3.11` instead of `python` in every command below.

### 2.3 Get the project folder

If you have Git:

```bash
git clone <REPOSITORY_URL> Punarvaani
cd Punarvaani
```

Replace `<REPOSITORY_URL>` with the link the team lead sends. If you received a ZIP, extract it, then move into it:

```bash
cd "C:\path\to\the\extracted\Punarvaani"
```

> `cd` means "change directory". If the path contains spaces, put it in double quotes.

Check you are in the right place:

```bash
dir
```

On macOS/Linux use `ls`. You should see `README.md`, `main.py`, `Data`, `Models`, `scripts`, `tests`.

### 2.4 Create the private Python box (`venv`)

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the script, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Windows Command Prompt (cmd)**

```bat
.venv\Scripts\activate.bat
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

You should now see `(.venv)` at the start of your terminal prompt. That means the private box is active.

### 2.5 Install the project dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This can take several minutes and downloads a lot.

> **Windows / CPU-only note:** the file `requirements.txt` contains a package called `triton`, which only works on Linux with NVIDIA GPUs. If installation fails on that line, open `requirements.txt` in a text editor and delete the `triton` line, save, then run `pip install -r requirements.txt` again.

### 2.6 Prove the installation works

```bash
python -m unittest discover -s tests -v
```

Expected ending:

```text
Ran <some number> tests in ...s
OK (skipped=1)
```

If you see `OK`, your setup is correct. The number may change as the team adds tests. The one skipped test is a slow end-to-end test; that is normal.

If you see `ModuleNotFoundError`, a package is missing. Install it with `pip install <package-name>` and run the tests again. Common ones are `mediapipe`, `opencv-python`, and `decord`.

---

## 3. Repository map (only the parts you need)

```text
Punarvaani/
├── README.md                  # project overview (long, optional reading)
├── TEAM_GUIDE.md              # this file
├── main.py                    # train / evaluate / infer commands
├── config.py                  # all command-line options and the Hindi sound list
├── inference.py               # Python function for a single video
├── face_cropper.py            # finds the face and crops the mouth
├── mouth_geometry.py          # mouth-box maths
│
├── Data/
│   ├── dataset.py             # reads <speaker>/<clip>.mp4 + .txt pairs
│   ├── hindi_g2p.py           # Devanagari spelling -> phonemes
│   ├── metrics.py             # CER / WER calculation
│   └── synthetic.py           # makes fake test data (no real faces)
│
├── Models/
│   ├── ML_VALLR.py            # Stage 1: video -> phonemes
│   └── Sarvam.py              # Stage 2: phonemes -> Hindi text
│
├── scripts/
│   ├── preprocess_lips.py     # batch mouth cropping
│   ├── preflight.py           # checks the dataset/adapter before training
│   └── make_synthetic_dataset.py
│
├── hindi_dataset/             # Teammate 1 puts data here
├── checkpoints/
│   └── sarvam_lora/           # Teammate 2 puts the adapter here
└── tests/                     # automated tests
```

**Golden rule:** do not rename or move any folder or file listed above. The code expects these exact names.

---

# Part 1 — Teammate 1: Hindi video dataset

## 1.1 What you deliver

A folder named `hindi_dataset/` with this exact shape:

```text
hindi_dataset/
├── spk01/
│   ├── spk01_001.mp4
│   ├── spk01_001.txt
│   ├── spk01_002.mp4
│   ├── spk01_002.txt
│   └── ...
├── spk02/
│   ├── spk02_001.mp4
│   ├── spk02_001.txt
│   └── ...
└── spk03/
    └── ...
```

Rules that the code enforces:

1. Every video is inside a **speaker folder** (`spk01`, `spk02`, ...). Do not put videos directly in `hindi_dataset/`.
2. Every video has a transcript with the **same base name** and a `.txt` extension.
3. There must be **at least two speaker folders**, so the validation set can contain a person the model has not seen during training.
4. Names should be ASCII, lowercase, and contain no spaces. Use `spk01_001`, not `Speaker One (final).mp4`.

## 1.2 How much data to collect

| Target | Clips | Speakers |
|---|---|---|
| Minimum prototype | 300–500 | 5–10 |
| Better research set | 1,500+ | 15+ |

Start with 5–10 speakers and 30–60 clips each. It is much better to have clean, consistent clips than many messy ones.

## 1.3 Recording rules

- **Camera angle:** straight in front of the face. The person looks directly at the camera, head level, no strong tilt or turn.
- **Lighting:** soft light on the whole face. Avoid a ceiling light directly above, which creates a shadow under the lower lip.
- **Framing:** head and shoulders visible; the mouth is never covered.
- **Obstructions:** no hands, microphone, scarf, mask, hair, or cup in front of the mouth.
- **Resolution:** 720p or higher.
- **Frame rate:** 25 or 30 frames per second.
- **Background:** quiet and still if possible. No other faces in the frame.
- **Clip length:** **1.5 to 3 seconds**. That is about 2–6 words or 10–25 syllables.
  - Very short clips (under 1 second) give the model too little visual context.
  - Very long clips (over 4 seconds) may exceed the model's time steps and be dropped.
- **Content:** everyday sentences. Useful categories: greetings, needs, pain/health, food/water, family, numbers (spelled out), directions, requests.
- **One speaker per folder.** Never mix two people in the same `spkXX` folder.

Example sentences (say them naturally, not robotically):

```text
नमस्ते भारत
मुझे पानी चाहिए
मैं ठीक हूँ
दर्द हो रहा है
डॉक्टर को बुलाओ
मुझे घर जाना है
यह कितना है
मेरा नाम राहुल है
कृपया मदद करें
मैं भूखा हूँ
```

## 1.4 Transcript rules

Write exactly what is spoken, in **Hindi Devanagari only**.

1. **No English letters or digits.**
   - Wrong: `School bus`, `2025`, `5 log`
   - Right: `स्कूल बस`, `दो हज़ार पच्चीस`, `पाँच लोग`
2. **Spell out numbers in Hindi words.**
   - Wrong: `२५`
   - Right: `पच्चीस`
3. **Write English loanwords in Devanagari.**
   - Wrong: `doctor`
   - Right: `डॉक्टर`
4. **No punctuation except normal spaces and the danda `।`.**
   - Remove commas, question marks, quotes, emojis, hyphens.
5. **Do not add explanations.** The transcript is only the spoken sentence.
6. **UTF-8 encoding.** When saving from Notepad, choose UTF-8. When saving from VS Code, the default UTF-8 is correct.
7. **One line per transcript.** Extra blank lines are ignored, but keep it to one sentence.

Examples:

| Spoken | Wrong `.txt` | Correct `.txt` |
|---|---|---|
| "I need water" | `Mujhe pani chahiye` | `मुझे पानी चाहिए` |
| "Call the doctor" | `Call doctor!` | `डॉक्टर को बुलाओ` |
| "It is 25 rupees" | `25 rupaye` | `पच्चीस रुपये` |
| "I am fine" | `I'm fine` | `मैं ठीक हूँ` |

## 1.5 Step-by-step: prepare and validate the dataset

### Step 1 — Put raw clips in a folder

Create a folder outside the repo, for example `raw_videos/`, with the same speaker-folder layout:

```text
raw_videos/
├── spk01/
├── spk02/
└── spk03/
```

Put the original `.mp4` files and matching `.txt` files there.

### Step 2 — Run the mouth cropper

The cropper finds the face, tracks the lips, and writes stabilized 224×224 mouth videos. It **keeps the speaker folders**.

```bash
python scripts/preprocess_lips.py --input_dir ./raw_videos --output_dir ./hindi_dataset --max_frames 96 --frame_size 224
```

What the options mean:

- `--input_dir`: your raw speaker folders.
- `--output_dir`: where the cleaned dataset will be written.
- `--max_frames 96`: read at most 96 frames per clip (about 3.8 seconds at 25 fps).
- `--frame_size 224`: output mouth video is 224×224 pixels.

Expected output:

```text
Found 60 video files in './raw_videos'. Processing...
Preprocessing: 100%|...|
Preprocessing Complete:
  Success: 60/60 (100.0%)
  Diagnostic Report saved to: hindi_dataset/preprocessing_report.json
```

Open `hindi_dataset/preprocessing_report.json`. If `crop_success_rate_pct` is below 95%, read the `failed_videos` reasons and fix those clips (usually the face was not visible, blurred, or sideways).

> If you see `MediaPipe FaceCropper is unavailable`, install it inside the activated environment:
>
> ```bash
> pip install mediapipe
> ```
>
> Then run the cropper again. If `pip` cannot find a version, you are probably on Python 3.13; create a Python 3.11 environment instead.

### Step 3 — Run the preflight check

This checks every pair, transcript, speaker folder, and CTC length before anyone spends GPU time.

```bash
python scripts/preflight.py --data_dir ./hindi_dataset --output ./preflight_report.json
```

By default, preflight assumes training will crop mouths on the fly (`--crop_mode mouth`), so it requires MediaPipe. If you already ran `preprocess_lips.py` and will train with `--preprocessed`, add `--crop_mode none`.

Expected ending:

```text
Preflight report written to .../preflight_report.json
  dataset: 60/60 pairs, 6 speakers, invalid CTC 0.0%
Preflight OK
```

What to look at in `preflight_report.json`:

- `pairs`: number of valid video+transcript pairs.
- `speakers`: one entry per speaker folder.
- `missing_transcripts`: videos with no `.txt`.
- `empty_transcripts`: `.txt` files with no text.
- `invalid_ctc`: clips whose transcript is too long for the number of frames.
- `non_devanagari`: transcripts containing English letters, digits, or symbols.
- `unsupported_g2p_chars`: characters the phoneme converter does not know.
- `blocking`: problems that must be fixed before training.

If `blocking` is not empty, fix the listed issues and run the command again. Do not send the dataset until `Preflight OK` appears.

### Step 4 — Run the unit tests

```bash
python -m unittest discover -s tests -v
```

This confirms the code on your machine is healthy. It must end with `OK`.

### Step 5 — Hand over the dataset

Send the `hindi_dataset/` folder through the team's shared drive (Google Drive, OneDrive, institutional storage). Include:

- `hindi_dataset/` with all `spkXX/` folders.
- `preflight_report.json`.
- `hindi_dataset/preprocessing_report.json`.
- A one-line message: number of speakers, number of clips, total minutes, and any known issues.

Do **not** upload videos to a public GitHub repository or a public cloud link.

## 1.6 Teammate 1 checklist

Before you say "done":

- [ ] Every video has a matching `.txt` with the same base name.
- [ ] Every file lives inside a `spkXX/` folder.
- [ ] At least two speaker folders exist.
- [ ] Transcripts are pure Devanagari, numbers spelled out, no English.
- [ ] Clips are 1.5–3 seconds and frontal.
- [ ] `preprocessing_report.json` shows at least 95% crop success.
- [ ] `preflight_report.json` shows `Preflight OK` and 0 blocking issues.
- [ ] `python -m unittest discover -s tests -v` ends with `OK`.
- [ ] The handover contains both reports.

## 1.7 Teammate 1 troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `No valid video/transcript pairs found` | A `.txt` is missing or names differ | Make the base names identical: `spk01_001.mp4` + `spk01_001.txt` |
| `clips are not inside a speaker subfolder` | Videos are directly in `hindi_dataset/` | Move each video into `spkXX/` |
| `fewer than 2 speaker folders` | Only one person recorded | Add another speaker folder, or ask the lead before using `--allow_unknown_speakers` |
| `invalid CTC samples ...` | Transcript too long for the clip | Shorten the sentence or add frames (up to `--max_frames 96`) |
| Crop success below 95% | Face turned, dark lighting, mouth covered | Re-record those clips following Section 1.3 |
| G2P warns about a character | English letter, digit, or symbol in `.txt` | Rewrite the transcript in pure Devanagari |
| `MediaPipe ... unavailable` | MediaPipe not installed; training now stops instead of silently using full frames | `pip install mediapipe`, or use `--preprocessed` only if the clips are already mouth-cropped |
| Video will not open | Unusual codec | Re-encode with `ffmpeg -i input.mp4 -c:v libx264 -pix_fmt yuv420p output.mp4` |
| `Permission denied` when writing output | `hindi_dataset/` is open in another program | Close it and run again |

---

# Part 2 — Teammate 2: Sarvam-1 LoRA adapter

## 2.1 What you deliver

A folder named `checkpoints/sarvam_lora/` containing:

```text
checkpoints/sarvam_lora/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer_config.json
├── special_tokens_map.json
└── tokenizer.json            (if your tokenizer saves one)
```

This adapter teaches Sarvam-1 to turn noisy phoneme sequences into correct Hindi sentences.

You do **not** need the video dataset for this role. You can start immediately using a Hindi text corpus.

## 2.2 How Stage 2 works

Stage 1 produces a string such as:

```text
N A M A S TD E | BH AA R A TD
```

Stage 2 receives that string and must return:

```text
नमस्ते भारत
```

Because many sounds look identical on the lips, Stage 1 will sometimes output a look-alike sound. Stage 2 uses Hindi grammar and vocabulary to fix it.

## 2.3 Prerequisites

Inside the activated environment (`(.venv)` visible), install the LLM packages:

```bash
pip install "peft>=0.12.0" "accelerate>=0.33.0" "datasets>=2.20.0" transformers
```

Check the GPU:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Expected: `CUDA available: True`. If it says `False`, this role needs a GPU machine (Colab, Kaggle, or a lab GPU). Training on CPU is not practical for a 2B model.

## 2.4 The exact prompt contract

Every training row and every inference prompt must use these four special tokens:

```python
TAGS = ["<S2S>", "<PHONEMES>", "</PHONEMES>", "<TEXT>"]
```

The prompt format is exactly:

```text
<S2S>
<PHONEMES>
PHONEME SEQUENCE HERE
</PHONEMES>
<TEXT>
```

The expected answer is the Devanagari sentence, with no extra text.

Before training, the tokenizer must register the tags and the model must resize its embeddings:

```python
tokenizer.add_special_tokens({"additional_special_tokens": TAGS})
model.resize_token_embeddings(len(tokenizer))
```

Each tag must become **one token**, not several subword pieces:

```python
assert len(tokenizer.encode("<S2S>", add_special_tokens=False)) == 1
```

The provided training function already does these steps. Do not change the tag strings.

## 2.5 Step-by-step: build training data

### Step 1 — Get Hindi text

Download a Hindi text corpus in plain text, one sentence per line. Good sources:

- AI4Bharat IndicCorp (Hindi portion)
- Hindi Wikipedia dumps
- OSCAR Hindi
- Any clean collection of Hindi sentences approved by the team

Aim for **50,000 to 200,000 sentences** for the first adapter. More is better, but quality and clean Devanagari matter more than raw size.

### Step 2 — Convert text to phonemes and add noise

Create a new file `scripts/make_sarvam_dataset.py` and paste this complete script:

```python
"""Build Sarvam-1 training JSONL from a Hindi text file.

Usage:
    python scripts/make_sarvam_dataset.py --corpus hindi_corpus.txt --out sarvam_data
"""

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Data.hindi_g2p import devanagari_to_phonemes

TAGS = ["<S2S>", "<PHONEMES>", "</PHONEMES>", "<TEXT>"]

# Sounds that look the same on the lips. Swapping inside a group simulates
# realistic Stage-1 mistakes.
VISEME_GROUPS = [
    ["P", "B", "M"],
    ["PH", "BH"],
    ["TD", "THD", "DD", "DHD", "N"],
    ["TR", "THR", "DR", "DHR", "NR"],
    ["K", "KH", "G", "GH", "NG"],
    ["CH", "CHH", "J", "JH", "NY"],
]
VISEME_SWAP = {}
for group in VISEME_GROUPS:
    for token in group:
        VISEME_SWAP[token] = group


def clean_text(line: str) -> str:
    """Keep only Devanagari letters and spaces."""
    line = re.sub(r"[^\u0900-\u097F\s]", " ", line)
    return re.sub(r"\s+", " ", line).strip()


def inject_noise(phonemes, rng, swap_p=0.10, drop_p=0.03, dup_p=0.03):
    """Simulate visual-speech errors in the phoneme stream."""
    out = []
    for token in phonemes:
        if token == "|":
            out.append(token)
            continue
        roll = rng.random()
        if roll < drop_p:
            continue                      # drop a phoneme
        if roll < drop_p + dup_p:
            out.extend([token, token])    # duplicate a phoneme
            continue
        if roll < drop_p + dup_p + swap_p and token in VISEME_SWAP:
            alternatives = [t for t in VISEME_SWAP[token] if t != token]
            if alternatives:
                out.append(rng.choice(alternatives))
                continue
        out.append(token)
    return out


def make_prompt(phoneme_string: str) -> str:
    return (
        f"{TAGS[0]}\n"
        f"{TAGS[1]}\n{phoneme_string}\n{TAGS[2]}\n"
        f"{TAGS[3]}\n"
    )


def build(corpus_path: str, out_dir: str, val_fraction: float = 0.02, seed: int = 13):
    rng = random.Random(seed)
    rows = []

    with open(corpus_path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = clean_text(line)
            words = text.split()
            if len(words) < 2 or len(words) > 30:
                continue
            phonemes = devanagari_to_phonemes(text)
            if len(phonemes) < 3:
                continue
            noisy = inject_noise(phonemes, rng)
            rows.append(
                {"prompt": make_prompt(" ".join(noisy)), "target": text}
            )

    if len(rows) < 100:
        raise SystemExit(
            f"Only {len(rows)} usable rows. Check that the corpus is Hindi Devanagari text."
        )

    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_fraction))
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "val.jsonl"), "w", encoding="utf-8") as handle:
        for row in rows[:n_val]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "train.jsonl"), "w", encoding="utf-8") as handle:
        for row in rows[n_val:]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Rows: {len(rows)} total, {n_val} validation, {len(rows) - n_val} train")
    print(f"Written to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="Plain Hindi text, one sentence per line")
    parser.add_argument("--out", default="sarvam_data", help="Output directory")
    parser.add_argument("--val_fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    build(args.corpus, args.out, args.val_fraction, args.seed)
```

Run it:

```bash
python scripts/make_sarvam_dataset.py --corpus hindi_corpus.txt --out sarvam_data
```

Expected:

```text
Rows: 73421 total, 1468 validation, 71953 train
Written to: .../sarvam_data
```

Inspect a few lines of `sarvam_data/train.jsonl`. Each line must look like this:

```json
{"prompt": "<S2S>\n<PHONEMES>\nN A M A S TD E | BH AA R A TD\n</PHONEMES>\n<TEXT>\n", "target": "नमस्ते भारत"}
```

Why noise matters: if you train only on perfect phonemes, the adapter will fail on real Stage-1 output, which contains mistakes. The 10% viseme swaps, 3% drops, and 3% duplications simulate those mistakes.

### Step 3 — Train the adapter

The repository already contains the training function. Run it from the project root:

```bash
python -c "from Models.Sarvam import train_sarvam_lora; train_sarvam_lora('sarvam_data/train.jsonl', 'sarvam_data/val.jsonl', 'checkpoints/sarvam_lora')"
```

What this does:

1. Loads tokenizer and base model `sarvamai/sarvam-1`.
2. Registers the four special tokens.
3. Resizes token embeddings.
4. Attaches a LoRA adapter.
5. Trains for 5 epochs.
6. Saves the adapter and tokenizer to `checkpoints/sarvam_lora/`.

If you need different settings, write a small Python file instead:

```python
from Models.Sarvam import train_sarvam_lora

train_sarvam_lora(
    train_jsonl_path="sarvam_data/train.jsonl",
    val_jsonl_path="sarvam_data/val.jsonl",
    output_dir="checkpoints/sarvam_lora",
    base_model_id="sarvamai/sarvam-1",
    epochs=3,
    batch_size=4,
    lr=2e-4,
)
```

Recommended hyperparameters:

| Setting | Value |
|---|---|
| Base model | `sarvamai/sarvam-1` |
| LoRA rank `r` | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Learning rate | `2e-4` |
| Epochs | 3–5 |
| Hardware | one GPU with at least 16 GB VRAM (T4, A10G, RTX 3090, A100) |

Training tips:

- If you run out of GPU memory, lower `batch_size` to 2 or 1.
- If training is unstable, lower the learning rate to `1e-4`.
- Keep one validation run (`val.jsonl`) so you can see whether it is learning.
- Do not manually edit the adapter files.

### Step 4 — Verify the adapter before handing it over

Run this from the project root:

```bash
python -c "from Models.Sarvam import validate_sarvam_adapter_compatibility as v; report = v('checkpoints/sarvam_lora'); print(report); assert report['is_compatible']"
```

Expected: `is_compatible: True`, and no assertion error.

For a stronger check that also tests the special tokens (only works if the adapter folder contains tokenizer files):

```bash
python -c "from transformers import AutoTokenizer; from Models.Sarvam import validate_sarvam_adapter_compatibility as v; t = AutoTokenizer.from_pretrained('checkpoints/sarvam_lora'); report = v('checkpoints/sarvam_lora', tokenizer=t); print(report); assert report['is_compatible']"
```

Then run the full preflight against the dataset, if the dataset is available:

```bash
python scripts/preflight.py --data_dir ./hindi_dataset --sarvam_adapter_path ./checkpoints/sarvam_lora
```

The adapter section must show `compatible=True`.

### Step 5 — Hand over the adapter

Send the whole `checkpoints/sarvam_lora/` folder through the team's shared drive. Also include:

- The `sarvam_data/` folder or a link to it (so training can be reproduced).
- A short note with: number of training rows, epochs, final training loss, and the verification output.
- The corpus source and licence, if the team needs to report it.

Do not upload the adapter to a public model hub.

## 2.6 Teammate 2 checklist

- [ ] Hindi corpus is clean Devanagari text.
- [ ] `sarvam_data/train.jsonl` and `sarvam_data/val.jsonl` exist.
- [ ] Every line has exactly two keys: `prompt` and `target`.
- [ ] Prompt starts with `<S2S>` and ends with `<TEXT>\n`.
- [ ] Targets are clean Devanagari (no English, no digits).
- [ ] Adapter folder has `adapter_config.json` and `adapter_model.safetensors` (or `.bin`).
- [ ] Adapter folder has tokenizer files.
- [ ] `validate_sarvam_adapter_compatibility` returns `is_compatible: True`.
- [ ] `preflight.py` reports the adapter as compatible.
- [ ] You wrote down rows, epochs, loss, corpus source.

## 2.7 Teammate 2 troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: peft` | Extra packages not installed | `pip install "peft>=0.12" accelerate datasets` |
| `CUDA available: False` | No GPU / wrong environment | Use a GPU machine; check `nvidia-smi` |
| `CUDA out of memory` | Batch too large | `batch_size=1` or `2`; shorter rows; gradient accumulation |
| `Only N usable rows` | Corpus not Hindi, or too short | Check the file is UTF-8 Devanagari, one sentence per line |
| `is_compatible` is False | Wrong base model or missing files | Re-export from the same `sarvamai/sarvam-1` training run |
| Preflight says base model mismatch | Adapter trained on a different model | Retrain with `base_model_id="sarvamai/sarvam-1"` |
| Preflight says missing weights | Only config was copied | Copy `adapter_model.safetensors` |
| Tokenizer splits `<S2S>` into pieces | Tags were not registered before resizing | Use the provided training function without modifying the tag code |
| Adapter loads but output is garbage | Too little data / no noise / too many epochs | Rebuild JSONL with the noise script; try 3 epochs |
| Training loss does not fall | Learning rate issue or bad data | Check JSONL; try `lr=1e-4` |

---

# Part 3 — Teammate 3: Train and evaluate Stage 1 (GPU owner)

If your team has a separate GPU/training owner, that person does this part. Otherwise the team lead handles it.

## 3.1 What you deliver

- `checkpoints/punarvaani_stage1.pth` — the trained Stage-1 model.
- A short `stage1_eval_report.txt` with CTC loss, phoneme error rate (PER), and CER/WER if Stage 2 is attached.
- The exact command used for training.

## 3.2 Check the GPU

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Expected: `True` and a GPU name. If `False`, fix the PyTorch/CUDA installation before training.

## 3.3 Validate the inputs first

```bash
python scripts/preflight.py --data_dir ./hindi_dataset --sarvam_adapter_path ./checkpoints/sarvam_lora --output ./preflight_report.json
```

Do not start a long training run until this says `Preflight OK`.

## 3.4 Train

```bash
python main.py --mode train --language hindi --data_dir ./hindi_dataset --save_model_path ./checkpoints/punarvaani_stage1.pth --max_frames 96 --batch_size 4 --epochs 25 --crop_mode mouth --device cuda
```

What the important options mean:

- `--data_dir`: folder with `spkXX/` clips and transcripts.
- `--save_model_path`: where the best checkpoint is written.
- `--max_frames 96`: maximum frames per clip.
- `--batch_size 4`: clips processed together. Lower this if the GPU runs out of memory.
- `--epochs 25`: number of full passes over the data.
- `--crop_mode mouth`: crop the mouth at training time. If the dataset is **already** mouth-cropped by `preprocess_lips.py`, add `--preprocessed` instead.
- `--device cuda`: use the GPU.

What a good run looks like:

```text
Epoch [1/25] | Train Loss: 5.1234 (PER: 95.2%, dropped 0) | Val Loss: 5.2040 (PER: 93.8%, dropped 0) | LR: 1.00e-04
...
  --> Saved new best checkpoint ... (Val PER: ...)
```

The loss and PER should generally go down. A flat line means something is wrong (wrong data, wrong crop, or learning rate).

If you run out of memory, try, in this order:

1. `--batch_size 2`
2. `--max_frames 64`
3. `--spatial_pool_tokens 5x5`
4. `--vit_depth 6`
5. `--vit_embed_dim 512`

Do not silently drop `--crop_mode mouth` unless the data is already cropped.

## 3.5 Evaluate on a held-out set

Put the test speakers in a separate folder, for example `hindi_dataset_test/`, with the same speaker-folder layout. Then:

```bash
python main.py --mode eval --data_dir ./hindi_dataset --val_dir ./hindi_dataset_test --save_model_path ./checkpoints/punarvaani_stage1.pth --sarvam_adapter_path ./checkpoints/sarvam_lora --eval_batch_size 4 --max_frames 96 --device cuda
```

Expected output:

```text
--- Stage 1 Evaluation Results ---
  CTC Loss: ...
  Phoneme Error Rate (PER): ...%
--- End-to-End Evaluation Results ---
  Character Error Rate (CER): ...%
  Word Error Rate (WER): ...%
```

If you see `[Stage 2] Sarvam adapter is not active (MOCK mode)`, the adapter did not load. Read the printed reason and return to Teammate 2.

## 3.6 Run one video

```bash
python main.py --mode infer --infer_video_path ./some_test_video.mp4 --save_model_path ./checkpoints/punarvaani_stage1.pth --sarvam_adapter_path ./checkpoints/sarvam_lora --crop_mode mouth --device cuda
```

Expected ending:

```text
 PREDICTED PHONEMES : ...
 FINAL HINDI TEXT   : ...
```

A `[MOCK TRANSCRIPTION: ...]` output means Stage 2 is not active.

---

# Part 4 — Teammate 4: Quality and handover (optional)

If someone owns quality, they run these checks after Teammates 1–3 hand over:

1. Unit tests:
   ```bash
   python -m unittest discover -s tests -v
   ```
   Must end with `OK`.

2. Synthetic end-to-end smoke test:
   ```bash
   python scripts/make_synthetic_dataset.py --out .tmp/synthetic --speakers 3 --clips 2
   python scripts/preflight.py --data_dir .tmp/synthetic --output .tmp/preflight.json
   ```
   Must print `Preflight OK`.

3. Real dataset preflight:
   ```bash
   python scripts/preflight.py --data_dir ./hindi_dataset --sarvam_adapter_path ./checkpoints/sarvam_lora
   ```
   Must print `Preflight OK`.

4. Adapter validation:
   ```bash
   python -c "from Models.Sarvam import validate_sarvam_adapter_compatibility as v; assert v('checkpoints/sarvam_lora')['is_compatible']"
   ```

5. Confirm no speaker leak: the loader enforces speaker-disjoint splits; check the dataset has at least two speakers and that the held-out set uses different speakers.

6. Confirm the handover folder structure (Section 5).

---

# Part 5 — Joint handover folders

Recommended final layout on the shared drive:

```text
Punarvaani_handover/
├── dataset/
│   ├── hindi_dataset/              # spkXX/ folders, cropped
│   ├── preprocessing_report.json
│   └── preflight_report.json
├── sarvam/
│   ├── sarvam_lora/                # adapter files
│   ├── sarvam_data/                # train.jsonl, val.jsonl
│   └── training_notes.txt          # rows, epochs, loss, corpus source
└── stage1/
    ├── punarvaani_stage1.pth
    └── stage1_eval_report.txt
```

Every handover message should state:

```text
Role:
What I built:
Folder/link:
Commands I ran:
Checks I ran and their result:
Known issues:
Who should use this next:
```

---

# Part 6 — All commands in one place

```bash
# setup (once)
python -m venv .venv
.venv\Scripts\Activate.ps1              # Windows PowerShell
# source .venv/bin/activate             # macOS / Linux
pip install -r requirements.txt

# health check
python -m unittest discover -s tests -v

# Teammate 1: crop + validate
python scripts/preprocess_lips.py --input_dir ./raw_videos --output_dir ./hindi_dataset --max_frames 96 --frame_size 224
python scripts/preflight.py --data_dir ./hindi_dataset --output ./preflight_report.json

# Teammate 2: build data + train + verify
python scripts/make_sarvam_dataset.py --corpus hindi_corpus.txt --out sarvam_data
python -c "from Models.Sarvam import train_sarvam_lora; train_sarvam_lora('sarvam_data/train.jsonl', 'sarvam_data/val.jsonl', 'checkpoints/sarvam_lora')"
python -c "from Models.Sarvam import validate_sarvam_adapter_compatibility as v; assert v('checkpoints/sarvam_lora')['is_compatible']"

# Teammate 3: train + evaluate + infer
python main.py --mode train --language hindi --data_dir ./hindi_dataset --save_model_path ./checkpoints/punarvaani_stage1.pth --max_frames 96 --batch_size 4 --epochs 25 --crop_mode mouth --device cuda
python main.py --mode eval --data_dir ./hindi_dataset --val_dir ./hindi_dataset_test --save_model_path ./checkpoints/punarvaani_stage1.pth --sarvam_adapter_path ./checkpoints/sarvam_lora --eval_batch_size 4 --max_frames 96 --device cuda
python main.py --mode infer --infer_video_path ./some_test_video.mp4 --save_model_path ./checkpoints/punarvaani_stage1.pth --sarvam_adapter_path ./checkpoints/sarvam_lora --crop_mode mouth --device cuda

# joint validation
python scripts/preflight.py --data_dir ./hindi_dataset --sarvam_adapter_path ./checkpoints/sarvam_lora
```

---

# Part 7 — Common errors and what they mean

| Error / message | Meaning | What to do |
|---|---|---|
| `python: command not found` | Python is not on PATH | Reinstall Python and tick "Add to PATH"; or use `py -3.11` |
| `Activate.ps1 cannot be loaded` | PowerShell blocks scripts | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `No module named torch` | Wrong environment | Activate `.venv`; run `pip install -r requirements.txt` |
| `No module named mediapipe` | Cropper dependency missing | `pip install mediapipe` |
| `No module named decord` | Video reader missing | `pip install decord`, or ignore: OpenCV fallback is used |
| `No valid video/transcript pairs` | Names do not match | `clip.mp4` must sit next to `clip.txt` |
| `not inside a speaker subfolder` | Files in the wrong place | Move into `spkXX/` |
| `fewer than 2 speaker folders` | Cannot make a fair validation split | Add a second speaker |
| `invalid CTC samples` | Transcript longer than the model's time steps | Shorten clip/sentence or increase `--max_frames` |
| `CUDA out of memory` | Model too large for GPU | Lower `--batch_size`, `--max_frames`, `--vit_depth` |
| `adapter base model ... does not match` | Adapter trained on another model | Retrain with `sarvamai/sarvam-1` |
| `MOCK mode` | Adapter not loaded | Check the printed reason; verify adapter files |
| `Preflight OK` but training is poor | Data quality, not code | Review crop success, transcripts, lighting |
| PowerShell shows `#< CLIXML` | Harmless PowerShell formatting | Ignore it |

---

# Part 8 — Privacy, safety, and rules

1. **Consent first.** Every volunteer must sign the team's consent form before recording.
2. **No public uploads.** Never upload faces, voices, transcripts, or checkpoints to a public GitHub repo or public cloud link. Use the team's private shared drive.
3. **No names in filenames.** Use IDs (`spk01`), not real names.
4. **Patient data only with approval.** Do not record or process patient videos until the team has ethics approval and a hospital partner.
5. **Do not edit code you do not understand.** If a command fails, send the full error message to the team lead.
6. **Do not rename folders** listed in Section 3. The code expects exact paths.
7. **Do not commit data or weights.** The repository `.gitignore` excludes videos, weights, and checkpoints for a reason.
8. **Keep your environment activated.** If `(.venv)` is missing from the prompt, run the activation command again.

---

# Part 9 — How to ask for help

When a command fails, send the team lead:

1. The exact command you ran.
2. The **full** error message (copy from the terminal, not a screenshot only).
3. Your operating system and Python version (`python --version`).
4. What you expected to happen.
5. The last successful step.

That information is usually enough to solve the problem in one reply.

---

## Final definition of done

The project handover is complete when all of these are true:

- [ ] `hindi_dataset/` passes `scripts/preflight.py` with 0 blocking issues.
- [ ] `checkpoints/sarvam_lora/` passes `validate_sarvam_adapter_compatibility` and preflight.
- [ ] Stage 1 checkpoint exists and evaluation produces PER/CER/WER numbers.
- [ ] `python -m unittest discover -s tests -v` ends with `OK`.
- [ ] One real video produces phonemes and Hindi text through both stages.
- [ ] Every handover message includes the folder, commands, checks, and known issues.

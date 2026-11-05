# Stage-1 Training Runbook — Windows / NVIDIA A5000

> **Who this is for:** the person who will run Stage-1 training on the college Windows PC. No prior
> machine-learning or terminal experience is assumed. Follow the steps in order and do not skip a
> gate.
>
> **What this covers:** cloning the repo, installing the environment, placing the dataset and
> adapter, the four gates before full training, the full training command, monitoring, evaluation,
> inference, and where the weights end up.
>
> **Related documents:** [`TEAM_GUIDE.md`](../TEAM_GUIDE.md) for dataset/adapter creation,
> [`ARCHITECTURE_PLAN.md`](ARCHITECTURE_PLAN.md) for the model design.

---

## 0. The whole process on one page

```text
1. Install Python 3.11 + Git
2. Clone the repo
3. Create .venv and install CUDA PyTorch + dependencies
4. GATE 1: run the unit test suite
5. Put data in speaker folders + adapter in its folder
6. GATE 2: preflight + visually inspect crops
7. GATE 3: overfit a tiny real dataset
8. GATE 4: train on a 20% subset for a few epochs
9. FULL TRAINING
10. Evaluate on held-out test speakers
11. Run one video end-to-end
12. Back up the checkpoint
```

You produce one main file:

```text
D:\punarvaani\models\stage1\punarvaani_stage1.pth
```

The Stage-2 adapter is separate:

```text
D:\punarvaani\models\sarvam_lora\
```

The rest of this document explains every step in detail.

---

## 1. Before you start: what you need

Have these ready:

- AnyDesk access to the college Windows PC.
- The GitHub repository URL.
- The dataset from Teammate 1 (`.mp4` + `.txt` pairs in speaker folders).
- The Sarvam-1 LoRA adapter from Teammate 2 (`adapter_config.json`, `adapter_model.safetensors`,
  tokenizer files).
- Enough free disk space. Video datasets are large; check before copying.
- About 24 GB of GPU memory (NVIDIA RTX A5000) and 128 GB of RAM.

Do **not** start if:

- The dataset has no speaker folders.
- The dataset has fewer than two speakers.
- The adapter is missing `adapter_config.json` or the weight file.
- The PC is on a metered/slow network drive. Copy data to a local SSD first.

---

## 2. Decide the folder layout

### 2.1 First, check whether the Desktop is OneDrive-synced

Open PowerShell and run:

```powershell
[Environment]::GetFolderPath("Desktop")
```

- If the output contains `OneDrive`, **do not store the dataset or checkpoints there**. OneDrive
  syncs and locks large files and makes training slow.
- Use a local drive such as `D:\punarvaani` or `C:\punarvaani`.

### 2.2 Recommended layout

Keep the git clone and the large artifacts separate:

```text
D:\punarvaani\
├── repo\                         # git clone of the project
│   ├── main.py
│   ├── Data\
│   ├── Models\
│   ├── scripts\
│   └── tests\
│
├── data\
│   ├── raw\
│   │   ├── train\
│   │   │   ├── spk01\
│   │   │   │   ├── spk01_001.mp4
│   │   │   │   └── spk01_001.txt
│   │   │   └── spk02\
│   │   ├── val\
│   │   │   └── spk11\...
│   │   └── test\
│   │       └── spk21\...
│   ├── train\                    # mouth-cropped output
│   │   ├── spk01\...
│   │   └── spk02\...
│   ├── val\
│   │   └── spk11\...
│   └── test\
│       └── spk21\...
│
├── models\
│   ├── sarvam_lora\
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── tokenizer_config.json
│   │   └── special_tokens_map.json
│   └── stage1\
│       ├── micro_overfit.pth
│       ├── trial_20pct.pth
│       └── punarvaani_stage1.pth
│
├── logs\
├── reports\
└── scratch\
```

Create it:

```powershell
mkdir D:\punarvaani
mkdir D:\punarvaani\data\raw\train, D:\punarvaani\data\raw\val, D:\punarvaani\data\raw\test
mkdir D:\punarvaani\data\train, D:\punarvaani\data\val, D:\punarvaani\data\test
mkdir D:\punarvaani\models\stage1, D:\punarvaani\models\sarvam_lora
mkdir D:\punarvaani\logs, D:\punarvaani\reports, D:\punarvaani\scratch
```

### 2.3 If you prefer to keep everything inside the repo

That also works. The repo already provides:

```text
repo\
├── hindi_dataset\spk01\...
├── hindi_val\spk11\...
├── hindi_test\spk21\...
├── checkpoints\sarvam_lora\...
├── checkpoints\punarvaani_stage1.pth
└── logs\
```

The `.gitignore` excludes these paths. If you choose this layout, replace every `D:\punarvaani\...`
path below with the matching path inside the repo.

> **Never run `git clean -xdf` or `git clean -fdX`** in the repo if your data or checkpoints live
> inside it. Those commands delete ignored files.

---

## 3. Install the software

### 3.1 Open PowerShell

Press the Windows key, type `PowerShell`, press Enter.

### 3.2 Check the GPU

```powershell
nvidia-smi
```

You should see `NVIDIA RTX A5000` and a CUDA version (12.x is ideal). If `nvidia-smi` is not
recognized, the NVIDIA driver is missing; ask the lab administrator to install it.

### 3.3 Check Python

```powershell
py -3.11 --version
```

Expected: `Python 3.11.x`.

If Python 3.11 is missing, install it from <https://www.python.org/downloads/> and tick
**"Add python.exe to PATH"**. Do not use Python 3.13: MediaPipe wheels lag behind.

### 3.4 Check Git

```powershell
git --version
```

If Git is missing, install it from <https://git-scm.com/downloads>.

---

## 4. Clone the repository

```powershell
cd D:\punarvaani
git clone <REPOSITORY_URL> repo
cd repo
```

Replace `<REPOSITORY_URL>` with the link the team lead sends. The clone folder is `repo` in this
guide; any name works as long as you use the same path in every command.

Verify you are in the right place:

```powershell
Get-ChildItem
```

You should see `main.py`, `config.py`, `Data`, `Models`, `scripts`, `tests`, `README.md`.

---

## 5. Create the Python environment

```powershell
cd D:\punarvaani\repo
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell refuses to run the activation script:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

You should now see `(.venv)` at the start of the prompt. Every command in this guide assumes the
environment is active. If it disappears, run the activation command again.

---

## 6. Install dependencies

### 6.1 Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 6.2 Install CUDA PyTorch first

This is important. If you install `requirements.txt` first, pip may pick a CPU-only PyTorch.

```powershell
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

### 6.3 Install the rest of the requirements (Windows-safe)

The checked-in `requirements.txt` contains Linux-only packages (`triton`, `nvidia-*`) and repeats
the PyTorch packages. Create a filtered copy for Windows:

```powershell
Get-Content requirements.txt | Where-Object {
    $_ -notmatch '^(triton|nvidia-|torch==|torchvision==|torchaudio==|decord==|mediapipe==)'
} | Set-Content requirements.windows.txt

pip install -r requirements.windows.txt
Remove-Item requirements.windows.txt
```

### 6.4 Install the LLM and video extras

```powershell
pip install peft accelerate datasets
pip install mediapipe
pip install decord
```

- `decord` may fail on Windows/Python 3.11. That is acceptable: the code falls back to OpenCV.
- `mediapipe` is required if you will crop mouths on the fly or run `preprocess_lips.py`. It is
  installed separately above so pip can resolve its NumPy requirement; a NumPy downgrade is fine.

### 6.5 Verify the installation

```powershell
python -c "import torch, mediapipe, peft; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Expected:

```text
torch 2.4.1+cu121
cuda 12.1
available True
NVIDIA RTX A5000
```

If `available` is `False`, stop. You installed a CPU wheel or the driver is wrong. Reinstall PyTorch
with the CUDA index URL above.

---

## 7. GATE 1 — Run the unit test suite

This proves the code and the environment work before any real data is touched.

```powershell
python -m unittest discover -s tests -v
```

Expected ending:

```text
Ran <some number> tests in ...s
OK (skipped=1)
```

The skipped test is a slow end-to-end test. It is normal.

If tests fail:

- Read the first failure, not the last.
- `ModuleNotFoundError` means a package is missing; install it inside the active `.venv`.
- Do not continue to training until the suite is green.

---

## 8. Place the data and adapter

### 8.1 Dataset layout

Every video must sit inside a speaker folder and have a matching `.txt`:

```text
data\train\
├── spk01\
│   ├── spk01_001.mp4
│   ├── spk01_001.txt
│   ├── spk01_002.mp4
│   └── spk01_002.txt
├── spk02\
│   ├── spk02_001.mp4
│   └── spk02_001.txt
└── spk03\
    └── ...
```

Rules:

- **Train:** at least two speaker folders.
- **Val/test:** at least one speaker folder (two or more is better; pass `--min_speakers 1` to preflight for these splits).
- Transcripts are UTF-8 Devanagari only. No English letters, no digits, no emojis.
- Clips are 1.5–3 s, frontal, mouth unobstructed.
- Do not put videos directly in `data\train\`; they must be inside `spkXX\`.
- Do not mix speakers in one folder.

### 8.2 Split train / val / test by speaker

If Teammate 1 gives you one folder with all speakers, split it **by speaker**, not by clip:

```text
data\raw\
├── train\spk01, spk02, spk03, ...
├── val\spk11, spk12
└── test\spk21, spk22
```

A speaker must appear in only one split. The code enforces this for `--val_split`, but if you
prepare the folders yourself, you must keep the splits disjoint manually.

If the dataset is small, use at least one validation speaker and one test speaker. Two or more is
better because a single speaker gives noisy metrics.

### 8.3 Adapter layout

```text
models\sarvam_lora\
├── adapter_config.json
├── adapter_model.safetensors      (or adapter_model.bin)
├── tokenizer_config.json
├── special_tokens_map.json
└── tokenizer.json                 (if exported)
```

Do not rename these files. The loader checks `adapter_config.json` and the weight file.

### 8.4 What not to do

- Do not train from a OneDrive or network folder.
- Do not put data inside the repo if the repo is on OneDrive.
- Do not merge train and test speakers.
- Do not edit the adapter files by hand.

---

## 9. Crop mouths (only if the clips are raw)

Skip this section if the dataset is already 224×224 mouth crops.

Run the cropper separately for each split so the speaker folders stay separated:

```powershell
python scripts/preprocess_lips.py --input_dir D:\punarvaani\data\raw\train --output_dir D:\punarvaani\data\train --max_frames 96 --frame_size 224
python scripts/preprocess_lips.py --input_dir D:\punarvaani\data\raw\val   --output_dir D:\punarvaani\data\val   --max_frames 96 --frame_size 224
python scripts/preprocess_lips.py --input_dir D:\punarvaani\data\raw\test  --output_dir D:\punarvaani\data\test  --max_frames 96 --frame_size 224
```

Expected:

```text
Preprocessing Complete:
  Success: 120/120 (100.0%)
  Diagnostic Report saved to: ...\preprocessing_report.json
```

Open each `preprocessing_report.json`. If `crop_success_rate_pct` is below 95%, inspect the
`failed_videos` reasons. Re-record or remove bad clips.

If the cropper says `MediaPipe FaceCropper is unavailable`, install it:

```powershell
pip install mediapipe
```

---

## 10. GATE 2 — Preflight and visual inspection

### 10.1 Run preflight

**If you cropped the clips with `preprocess_lips.py`:**

```powershell
python scripts/preflight.py --data_dir D:\punarvaani\data\train --sarvam_adapter_path D:\punarvaani\models\sarvam_lora --crop_mode none --output D:\punarvaani\reports\preflight_train.json
```

**If the clips are raw and training will crop on the fly:**

```powershell
python scripts/preflight.py --data_dir D:\punarvaani\data\train --sarvam_adapter_path D:\punarvaani\models\sarvam_lora --crop_mode mouth --output D:\punarvaani\reports\preflight_train.json
```

Expected ending:

```text
Preflight report written to ...\preflight_train.json
  dataset: 120/120 pairs, 6 speakers, invalid CTC 0.0%
  adapter: compatible=True path=...\models\sarvam_lora
  runtime: python 3.11.x, CUDA=True, GPU=NVIDIA RTX A5000, disk_free=...
Preflight OK
```

`Preflight OK` means there are no blocking issues. Read the warnings too.

Run preflight on validation and test as well:

```powershell
python scripts/preflight.py --data_dir D:\punarvaani\data\val --crop_mode none --min_speakers 1 --output D:\punarvaani\reports\preflight_val.json
python scripts/preflight.py --data_dir D:\punarvaani\data\test --crop_mode none --min_speakers 1 --output D:\punarvaani\reports\preflight_test.json
```

### 10.2 What to check in the report

| Field | What it means | Action |
|---|---|---|
| `pairs` | Valid video+transcript pairs | Must equal the number of videos |
| `speakers` | Speaker folders found | At least 2 for train |
| `missing_transcripts` | Videos with no `.txt` | Add the transcript or remove the video |
| `empty_transcripts` | Empty `.txt` files | Fix or remove |
| `invalid_ctc` | Transcript too long for the clip | Shorten the sentence or raise `--max_frames` |
| `non_devanagari` | English/digits/symbols in transcripts | Rewrite in pure Devanagari |
| `unsupported_g2p_chars` | Characters the phoneme converter cannot map | Fix the transcript |
| `blocking` | Must-fix issues | Fix all of them before continuing |
| `warnings` | Non-fatal notes (e.g. missing decord) | Read and decide |
| `runtime.dependencies` | Installed/missing packages | Install `mediapipe` and `peft` |
| `runtime.cuda_available` | GPU availability | Must be `True` for training |

### 10.3 Inspect 10 clips and 10 crops by eye

Open random files from `data\train\`:

- Does each video show a **mouth crop**, not a whole face?
- Is the mouth centered and not cut off?
- Is the transcript matching what is spoken?
- Is the face frontal and well lit?

If the crops look wrong, stop. No amount of training fixes bad input. Re-run preprocessing or fix
the raw videos.

---

## 11. GATE 3 — Overfit a tiny real dataset

The goal is to prove the model **can learn**. A model that cannot memorize four clips has a data,
label, or pipeline bug.

### 11.1 Build the micro dataset

Create a folder with two speakers and two clips each, using **already-cropped** clips if possible:

```text
D:\punarvaani\data\micro\
├── spk01\
│   ├── spk01_001.mp4
│   ├── spk01_001.txt
│   ├── spk01_002.mp4
│   └── spk01_002.txt
└── spk02\
    ├── spk02_001.mp4
    ├── spk02_001.txt
    ├── spk02_002.mp4
    └── spk02_002.txt
```

Copy them from `data\train`. Use short, simple sentences.

### 11.2 Run the overfit test

If the micro clips are cropped:

```powershell
python main.py --mode train --data_dir D:\punarvaani\data\micro --val_dir D:\punarvaani\data\micro --save_model_path D:\punarvaani\models\stage1\micro_overfit.pth --val_split 0 --preprocessed --max_frames 48 --batch_size 2 --eval_batch_size 2 --epochs 80 --warmup_steps 20 --temporal_upsample_factor 2 --vit_embed_dim 128 --vit_depth 2 --vit_heads 4 --spatial_pool_tokens 4x4 --device cuda
```

If the micro clips are raw:

```powershell
python main.py --mode train --data_dir D:\punarvaani\data\micro --val_dir D:\punarvaani\data\micro --save_model_path D:\punarvaani\models\stage1\micro_overfit.pth --val_split 0 --crop_mode mouth --max_frames 48 --batch_size 2 --eval_batch_size 2 --epochs 80 --warmup_steps 20 --temporal_upsample_factor 2 --vit_embed_dim 128 --vit_depth 2 --vit_heads 4 --spatial_pool_tokens 4x4 --device cuda
```

### 11.3 Pass criteria

By the end of the run:

- Train loss has fallen from about `4.0` to **below ~1.0**.
- Train PER has fallen **below ~30%**.
- Validation loss/PER also improve, because validation is the same micro data.
- `dropped` is `0`.

### 11.4 If it fails

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss stays near 4.0 | Model is not learning | Check LR/warmup, labels, crops |
| Loss becomes `NaN` | Instability or bad data | Stop; verify transcripts and crops; restart with a longer warmup. The base LR is fixed at 1e-4 in code. |
| `dropped` > 0 | Transcripts too long for `T_ctc` | Raise `--max_frames`, shorten sentences |
| PER stays 100% | Model predicts blank everywhere | Check crops and transcripts; try more epochs |
| `RuntimeError: crop_mode ... requires MediaPipe` | MediaPipe missing | Install it, or use `--preprocessed` only if clips are cropped |
| `No valid video/transcript pairs` | Names/paths wrong | Check `clip.mp4` + `clip.txt` pairs |

Do not continue until the overfit test passes.

---

## 12. GATE 4 — Train on a 20% subset

This answers: “Does the model learn on real, varied data, not just four memorized clips?”

```powershell
python main.py --mode train --data_dir D:\punarvaani\data\train --val_dir D:\punarvaani\data\val --save_model_path D:\punarvaani\models\stage1\trial_20pct.pth --val_split 0 --sample_size 0.2 --preprocessed --max_frames 96 --batch_size 4 --eval_batch_size 4 --epochs 5 --warmup_steps 200 --device cuda
```

If the data is raw, replace `--preprocessed` with `--crop_mode mouth`.

### Pass criteria

- Train loss decreases over the 5 epochs.
- Validation loss decreases at least initially.
- `dropped` is 0 or very small.
- No `NaN`/`inf` loss.

If it does not learn, do not launch the full run. Fix the data or training setup first.

### Reading the trial output

```text
Epoch [1/5] | Train Loss: 5.1234 (PER: 95.2%, dropped 0) | Val Loss: 5.2040 (PER: 93.8%, dropped 0) | LR: 1.00e-04
Epoch [2/5] | Train Loss: 4.4310 (PER: 88.1%, dropped 0) | Val Loss: 4.6120 (PER: 90.2%, dropped 0) | LR: 1.00e-04
...
```

- Loss should trend down.
- PER should trend down.
- A flat loss line means something is wrong.
- A growing gap between train and validation PER means overfitting (expected later, not in a 20% trial).

---

## 13. Full training

### 13.1 Command

If the splits are pre-made and the clips are cropped:

```powershell
python main.py --mode train --language hindi --data_dir D:\punarvaani\data\train --val_dir D:\punarvaani\data\val --save_model_path D:\punarvaani\models\stage1\punarvaani_stage1.pth --val_split 0 --preprocessed --max_frames 96 --batch_size 4 --eval_batch_size 4 --epochs 25 --warmup_steps 500 --strict_ctc --device cuda
```

If the data is raw, replace `--preprocessed` with `--crop_mode mouth`.

If you have one folder and no separate validation folder:

```powershell
python main.py --mode train --language hindi --data_dir D:\punarvaani\data\all --save_model_path D:\punarvaani\models\stage1\punarvaani_stage1.pth --val_split 0.2 --preprocessed --max_frames 96 --batch_size 4 --eval_batch_size 4 --epochs 25 --warmup_steps 500 --strict_ctc --device cuda
```

Important flags:

| Flag | Why |
|---|---|
| `--val_split 0` | Use all pre-split training speakers; validation is a separate folder |
| `--preprocessed` | Clips are already mouth-cropped |
| `--crop_mode mouth` | Raw clips: crop mouths during training (needs MediaPipe) |
| `--max_frames 96` | At most 96 frames per clip |
| `--batch_size 4` | Good starting point for an A5000 |
| `--strict_ctc` | Stop early if too many samples violate CTC length |
| `--device cuda` | Use the GPU |

### 13.2 Save the console output

```powershell
New-Item -ItemType Directory -Force D:\punarvaani\logs | Out-Null
$log = "D:\punarvaani\logs\train_$(Get-Date -Format yyyyMMdd_HHmmss).log"
python main.py ... 2>&1 | Tee-Object -FilePath $log
```

`logs/` is gitignored. Do not commit logs.

### 13.3 Run it detached (survives closing the terminal)

```powershell
New-Item -ItemType Directory -Force D:\punarvaani\logs | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Start-Process -FilePath "D:\punarvaani\repo\.venv\Scripts\python.exe" -ArgumentList @(
  "main.py","--mode","train",
  "--language","hindi",
  "--data_dir","D:\punarvaani\data\train",
  "--val_dir","D:\punarvaani\data\val",
  "--save_model_path","D:\punarvaani\models\stage1\punarvaani_stage1.pth",
  "--val_split","0",
  "--preprocessed",
  "--max_frames","96",
  "--batch_size","4",
  "--eval_batch_size","4",
  "--epochs","25",
  "--warmup_steps","500",
  "--strict_ctc",
  "--device","cuda"
) -RedirectStandardOutput "D:\punarvaani\logs\train_$ts.out" -RedirectStandardError "D:\punarvaani\logs\train_$ts.err" -WindowStyle Hidden
```

Check it is running:

```powershell
Get-Process python
Get-Content D:\punarvaani\logs\train_*.out -Wait -Tail 20
```

### 13.4 Keep the PC awake

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 15
```

- Keep the Windows user logged in. Locking the screen is fine; signing out kills the process.
- Disable Windows Update automatic restarts in Settings.

### 13.5 Memory fallback order

If CUDA runs out of memory, change one thing at a time:

1. `--batch_size 2`
2. `--max_frames 64`
3. `--spatial_pool_tokens 5x5`
4. `--vit_depth 6`
5. `--vit_embed_dim 512`

Do not change `--temporal_upsample_factor` unless you understand the CTC length effect.

---

## 14. Monitor the run

Open a second PowerShell window:

```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv -l 5
```

Watch the epoch lines in the log:

```text
Epoch [3/25] | Train Loss: 3.8120 (PER: 78.4%, dropped 0) | Val Loss: 4.1050 (PER: 84.1%, dropped 0) | LR: 1.00e-04
  --> Saved new best checkpoint to ...\punarvaani_stage1.pth (Val PER: 84.1%)
```

What each number means:

- **Train Loss / Val Loss** — average CTC loss for that epoch.
- **Train PER / Val PER** — phoneme error rate after greedy decoding.
- **dropped** — samples filtered because `T_ctc < U + R`. Should be 0 or tiny.
- **LR** — current learning rate (warmup then plateau decay).
- **Saved new best checkpoint** — validation PER improved; the checkpoint was overwritten.

Healthy run:

- Both losses trend down.
- Validation PER trends down.
- The train/validation gap stays small early; it may grow later as the model overfits.

Underfitting:

- Both train and validation loss stay high and flat.
- Fix: more epochs, better data/crops, check labels, check `dropped`.

Overfitting:

- Train loss keeps falling, validation loss rises or validation PER stalls/worsens.
- Fix: keep the best checkpoint, collect more data, stop the run, use a smaller model.

PER above 100% early is normal (insertions). Watch the trend.

---

## 15. Where the weights are stored

The full-run checkpoint is exactly where you passed `--save_model_path`:

```text
D:\punarvaani\models\stage1\punarvaani_stage1.pth
```

Facts:

- It is a PyTorch `state_dict` (weights only).
- It is overwritten every time validation PER improves. It is the **best** checkpoint, not
  necessarily the final epoch.
- It does **not** include optimizer state, scheduler state, or the epoch number. There is no exact
  resume-from-crash. If the run dies, restart.
- The file is gitignored. Back it up to the team's private shared drive after training.

Trial and gate checkpoints:

```text
D:\punarvaani\models\stage1\micro_overfit.pth
D:\punarvaani\models\stage1\trial_20pct.pth
D:\punarvaani\models\stage1\punarvaani_stage1.pth
```

You do not need to delete the gate checkpoints before full training. Each run starts from random
initialization and writes only to its own `--save_model_path`.

The Stage-2 adapter is separate:

```text
D:\punarvaani\models\sarvam_lora\
```

---

## 16. Evaluate on the held-out test set

Use the test folder only now. Never use it for tuning.

```powershell
python main.py --mode eval --data_dir D:\punarvaani\data\train --val_dir D:\punarvaani\data\test --save_model_path D:\punarvaani\models\stage1\punarvaani_stage1.pth --sarvam_adapter_path D:\punarvaani\models\sarvam_lora --preprocessed --max_frames 96 --eval_batch_size 4 --device cuda
```

If the data is raw, replace `--preprocessed` with `--crop_mode mouth`.

Expected output:

```text
--- Stage 1 Evaluation Results ---
  CTC Loss: ...
  Phoneme Error Rate (PER): ...%
  Dropped samples (CTC length): ...

--- End-to-End Evaluation Results ---
  Character Error Rate (CER): ...%
  Word Error Rate (WER): ...%
```

If you see `[Stage 2] Sarvam adapter is not active (MOCK mode)`, the adapter is not loading. Read
the reason and fix the adapter before trusting CER/WER.

Save this output to `D:\punarvaani\reports\stage1_eval_report.txt`.

---

## 17. Run one video end-to-end

```powershell
python main.py --mode infer --infer_video_path D:\punarvaani\scratch\test_video.mp4 --save_model_path D:\punarvaani\models\stage1\punarvaani_stage1.pth --sarvam_adapter_path D:\punarvaani\models\sarvam_lora --crop_mode mouth --max_frames 96 --device cuda
```

If the video is already a mouth crop:

```powershell
python main.py --mode infer --infer_video_path D:\punarvaani\scratch\test_video.mp4 --save_model_path D:\punarvaani\models\stage1\punarvaani_stage1.pth --sarvam_adapter_path D:\punarvaani\models\sarvam_lora --crop_mode none --max_frames 96 --device cuda
```

Expected ending:

```text
 PREDICTED PHONEMES : ...
 FINAL HINDI TEXT   : ...
```

A `[MOCK TRANSCRIPTION: ...]` means Stage 2 is not active.

---

## 18. What to hand back

Send the team lead:

1. `D:\punarvaani\models\stage1\punarvaani_stage1.pth`
2. `D:\punarvaani\reports\stage1_eval_report.txt`
3. The exact training command used.
4. The contents of the full training log (or the first and last epochs).
5. The final evaluation numbers: PER, and CER/WER if the adapter was active.
6. Any warnings or failures you saw.

Do not send the entire dataset or logs through chat. Use the team's private shared drive.

---

## 19. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `python: command not found` | Python not on PATH | Reinstall Python 3.11, tick "Add to PATH", or use `py -3.11` |
| `Activate.ps1 cannot be loaded` | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `No module named torch` | venv not active | `.\.venv\Scripts\Activate.ps1` |
| `torch.cuda.is_available()` is `False` | CPU wheel or driver issue | Reinstall PyTorch with the `cu121` index URL; run `nvidia-smi` |
| `No module named mediapipe` | MediaPipe missing | `pip install mediapipe` |
| `No module named peft` | PEFT missing | `pip install peft` |
| `No module named decord` | Optional video reader missing | `pip install decord`, or ignore (OpenCV fallback) |
| `crop_mode 'mouth' requires MediaPipe` | Training on raw clips without MediaPipe | Install MediaPipe, or use `--preprocessed` only if clips are cropped |
| `No valid video/transcript pairs` | File names do not match | `clip.mp4` must have `clip.txt` with the same base name |
| `not inside a speaker subfolder` | Videos in the wrong place | Move each video into `spkXX\` |
| `fewer than 2 speaker folders` | Not enough speakers | Add speakers; do not use `--allow_unknown_speakers` for real training |
| `invalid CTC samples` | Transcript longer than the model's time steps | Shorten sentences or raise `--max_frames` |
| `CUDA out of memory` | Batch/model too large | `--batch_size 2`, `--max_frames 64`, `--spatial_pool_tokens 5x5` |
| Loss is `NaN` | Instability or bad data | Fix data; restart; lower effective LR by increasing warmup |
| Loss flat at ~4.0 | Not learning | Check crops, transcripts, `dropped`, and that the GPU is used |
| `MOCK mode` in evaluation | Adapter missing or incompatible | Check `adapter_config.json`, weights, base model, tokenizer files |
| `adapter base model ... does not match` | Wrong adapter | Use an adapter trained from `sarvamai/sarvam-1` |
| `Preflight OK` but training poor | Data quality | Inspect crops and transcripts; more/better data |
| Very slow epochs | Data on HDD/OneDrive, low workers | Move to SSD, `--num_workers 8`, pre-crop, use decord |
| Low GPU utilization | CPU decoding bottleneck | Pre-crop, more workers, faster disk |
| AnyDesk disconnect kills training | Process tied to session | Use `Start-Process`, keep the user logged in, disable sleep |
| `#< CLIXML` in PowerShell | Harmless formatting | Ignore |
| `git clean` deleted data | Data was inside the repo | Keep data outside, or never use `git clean -xdf` |

---

## 20. Final checklist

Before marking the training task done:

- [ ] Python 3.11 environment is active.
- [ ] `torch.cuda.is_available()` is `True` and the GPU is an A5000.
- [ ] `python -m unittest discover -s tests -v` ends with `OK`.
- [ ] Dataset has speaker folders, pairs, pure Devanagari transcripts.
- [ ] Train/val/test speakers are disjoint.
- [ ] Preflight says `Preflight OK` for all splits.
- [ ] At least 10 mouth crops were inspected by eye.
- [ ] Overfit gate passed: loss < ~1.0 and PER < ~30%.
- [ ] 20% trial passed: loss trended down.
- [ ] Full training completed and saved the best checkpoint.
- [ ] Test-set evaluation produced PER (and CER/WER if the adapter loaded).
- [ ] One real video produced phonemes and Hindi text.
- [ ] `punarvaani_stage1.pth` backed up to the team's shared drive.
- [ ] Training command and logs saved.

---

## 21. Glossary

| Term | Meaning |
|---|---|
| **Epoch** | One full pass over the training data. |
| **Batch** | A group of clips processed together. |
| **Loss** | How wrong the model is; lower is better. |
| **PER** | Phoneme Error Rate; percentage of phonemes decoded incorrectly. |
| **CTC** | The training objective that aligns variable-length video to phoneme labels. |
| **Checkpoint** | A saved file of model weights (`.pth`). |
| **Best checkpoint** | The checkpoint with the lowest validation PER so far. |
| **Overfitting** | The model memorizes training speakers; validation gets worse. |
| **Underfitting** | The model fails to learn even the training data. |
| **Validation set** | Speakers the model does not train on, used to choose the best checkpoint. |
| **Test set** | Speakers used only once, for the final reported numbers. |
| **Adapter** | A small set of extra weights for the Stage-2 LLM. |
| **MOCK mode** | Stage 2 is disabled because the adapter is missing/incompatible. |
| **Dropped** | Samples skipped because the transcript is too long for the model's time steps. |

---

## 22. Quick command appendix

```powershell
# Environment
cd D:\punarvaani\repo
.\.venv\Scripts\Activate.ps1

# Gate 1
python -m unittest discover -s tests -v

# Gate 2
python scripts/preflight.py --data_dir D:\punarvaani\data\train --sarvam_adapter_path D:\punarvaani\models\sarvam_lora --crop_mode none --output D:\punarvaani\reports\preflight_train.json

# Gate 3 (cropped micro data)
python main.py --mode train --data_dir D:\punarvaani\data\micro --val_dir D:\punarvaani\data\micro --save_model_path D:\punarvaani\models\stage1\micro_overfit.pth --val_split 0 --preprocessed --max_frames 48 --batch_size 2 --eval_batch_size 2 --epochs 80 --warmup_steps 20 --temporal_upsample_factor 2 --vit_embed_dim 128 --vit_depth 2 --vit_heads 4 --spatial_pool_tokens 4x4 --device cuda

# Gate 4 (20% trial)
python main.py --mode train --data_dir D:\punarvaani\data\train --val_dir D:\punarvaani\data\val --save_model_path D:\punarvaani\models\stage1\trial_20pct.pth --val_split 0 --sample_size 0.2 --preprocessed --max_frames 96 --batch_size 4 --eval_batch_size 4 --epochs 5 --warmup_steps 200 --device cuda

# Full training
python main.py --mode train --language hindi --data_dir D:\punarvaani\data\train --val_dir D:\punarvaani\data\val --save_model_path D:\punarvaani\models\stage1\punarvaani_stage1.pth --val_split 0 --preprocessed --max_frames 96 --batch_size 4 --eval_batch_size 4 --epochs 25 --warmup_steps 500 --strict_ctc --device cuda

# Evaluation
python main.py --mode eval --data_dir D:\punarvaani\data\train --val_dir D:\punarvaani\data\test --save_model_path D:\punarvaani\models\stage1\punarvaani_stage1.pth --sarvam_adapter_path D:\punarvaani\models\sarvam_lora --preprocessed --max_frames 96 --eval_batch_size 4 --device cuda

# Inference
python main.py --mode infer --infer_video_path D:\punarvaani\scratch\test_video.mp4 --save_model_path D:\punarvaani\models\stage1\punarvaani_stage1.pth --sarvam_adapter_path D:\punarvaani\models\sarvam_lora --crop_mode mouth --max_frames 96 --device cuda
```

---

## 23. One rule to remember

**Do not launch the full run until the overfit gate passes.** A full run on broken data produces a
checkpoint that looks trained and is useless. The four gates exist to catch that before you spend
GPU time.

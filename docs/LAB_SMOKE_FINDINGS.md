# Punarvaani — Lab Smoke Test Findings

> Machine: college GPU PC (Windows, NVIDIA RTX A5000 24 GB, driver 596.51).
> Date: 2026-11-23. Companion to `docs/LAB_HANDOFF.md`.
> Scope: plumbing / pipeline sanity for the delivered smoke dataset. **Not** an accuracy claim.

## 1. TL;DR

- Environment is built and working: Python 3.11.9 venv, CUDA PyTorch 2.4.1 (A5000 visible), MediaPipe 0.10.14, PEFT/transformers, decord.
- Gate 1 (unit suite): **73 tests, OK (skipped=2)**.
- Dataset verified: **2,869 video/transcript pairs, 293 speakers, 1.502 h**, speaker-disjoint splits (train 220 / val 47 / test 26).
- Mouth cropping: micro **8/8**, val **316/316**, test **324/324**, train **2,229/2,229** — all **100%**, 0 failed.
- Preflight: micro **OK**, val **OK**, test **OK** (0% invalid CTC). Non-blocking warning: the danda `।` appears in a few transcripts and is an unsupported G2P char.
- **Important finding:** the documented micro-overfit recipe *fails on real clips* because `ReduceLROnPlateau` decays LR to ~5e-8. A no-decay LR floor + more epochs makes it pass (7/8 clips decode exactly correct). Code fix added: `--min_lr`, `--lr_patience`.
- Stage-2 adapter is **not delivered yet** (`checkpoints/sarvam_lora/` has only README), so Stage 2 is **MOCK**. No CER/WER.

## 2. Machine / environment state

Default `python` is **3.14.3** (no torch), so a 3.11 venv was created off C: (C: had only 3.1 GB free).

| Item | Value |
|---|---|
| Repo | `C:\Users\PRO-LAB-2\Desktop\Punarvaani\punarvaani` (branch `main`, clean, tip `40da2b4`) |
| Venv | `E:\punarvaani_venv` (Python 3.11.9) |
| torch | 2.4.1+cu121, `cuda.is_available() == True`, `NVIDIA RTX A5000` |
| numpy / cv2 | 2.0.2 / 4.10.0 |
| mediapipe | 0.10.14 (see pitfall below) |
| transformers / peft / accelerate / datasets | 4.44.2 / 0.13.2 / 1.15.0 / 4.1.1 |
| decord | 0.6.0 (installed successfully) |
| Disk | C: 3.1 GB free, D: 39.9 GB, E: ~146 GB — all data/venv placed on **E:** |

### Pitfalls hit

1. **`pip install mediapipe` grabbed 1.2.3**, which ships only the new `tasks` API and has **no `mp.solutions`**. `face_cropper.py` needs `mp.solutions.face_detection/face_mesh`. Fixed by pinning `mediapipe==0.10.14`.
2. **`scripts/preprocess_lips.py` cannot import `face_cropper`** when run as `python scripts\preprocess_lips.py`, because `sys.path[0]` becomes `scripts\`, not the repo root. Workaround: set `PYTHONPATH=<repo root>`. (Latent repo bug; not patched.)
3. **DataLoader `num_workers>0` crashes on Windows** as a background job (spawn/multiprocessing, early exit). Use `--num_workers 0` for reliable runs (also avoids a busy CPU during preprocessing).
4. **decord is used first** by `Data/dataset.py`; it reads the `mp4v` crop files fine, but returns **RGB** while OpenCV returns BGR. Not a correctness blocker here (training is consistent), just noted.

## 3. Dataset delivered

Source zip: `C:\Users\PRO-LAB-2\Desktop\punarvaani_smoke_v1.zip` (1.2 GB uncompressed) → extracted to `E:\punarvaani_data\raw`.

```text
raw/
├── raw_videos/<spkXX>/<clip>.mp4 + <clip>.txt   # 293 speakers, 2869 pairs
├── manifest.csv        # per-clip ASR/data metadata
├── report.json         # 3071 candidates, 2871 accepted, 1.502 h, 294 speakers
├── speakers.json       # spkID -> human name
├── splits.json         # speaker-disjoint train/val/test
└── identity/           # clips.csv, embeddings.npz, persons.json (unused by pipeline)
```

- Clips are **raw 1280×720, 25 fps, ~1.6 s (~40 frames)** — **not** mouth-cropped. Crop step was required.
- Every video has a matching `.txt`; transcripts are pure Devanagari (e.g. `कि राहूल गांधी पर धक्का देने`).
- Splits: **train 220 speakers / 2,229 clips**, **val 47 / 316**, **test 26 / 324**. No overlap, no missing speaker dirs.

Working copy built with hardlinks (no extra space): `E:\punarvaani_data\raw_split\{train,val,test}`.
Micro-overfit set: `E:\punarvaani_data\raw_micro` (spk38 + spk279, 4 shortest clips each).

## 4. Preprocessing (mouth crop, 224×224)

Command pattern (PYTHONPATH is required, see pitfall 2):

```powershell
$env:PYTHONPATH='C:\Users\PRO-LAB-2\Desktop\Punarvaani\punarvaani'
python scripts\preprocess_lips.py --input_dir <raw_split>\<split> --output_dir <cropped>\<split> --max_frames 96 --frame_size 224
```

| Split | Result | Output |
|---|---|---|
| micro | 8/8 (100%) | `E:\punarvaani_data\cropped_micro` |
| val | 316/316 (100%) | `E:\punarvaani_data\cropped\val` |
| test | 324/324 (100%) | `E:\punarvaani_data\cropped\test` |
| train | 2,229/2,229 (100%), 0 failed | `E:\punarvaani_data\cropped\train` |

Visual spot-check: crops are tight, centered mouth ROIs (224×224), stable across frames. Good.

## 5. Gate 1 — unit tests

`python -m unittest discover -s tests` → **Ran 73 tests, OK (skipped=2)**.
(The 2 skips are the `RUN_SLOW=1` E2E test(s).) Still green after the scheduler change.

## 6. Preflight

| Data | Result |
|---|---|
| `cropped_micro` | `Preflight OK` — 8/8 pairs, 2 speakers, 0% invalid CTC |
| `cropped/val` | `Preflight OK` — 316/316 pairs, 47 speakers, 0% invalid CTC |
| `cropped/test` | `Preflight OK` — 324/324 pairs, 26 speakers, 0% invalid CTC |
| `cropped/train` | pending (crop now complete at 2,229/2,229; preflight not yet run) |

Warning (non-blocking) on val/test: `unsupported G2P characters: {'।': N}` — a few transcripts contain the danda `।`, which the G2P drops. Minor label noise, not blocking.

## 7. Micro-overfit gate — KEY FINDING

### The documented recipe fails on real clips

`docs/LAB_HANDOFF.md` §7.4 command (tiny model, `epochs 80`, `warmup_steps 20`, default `ReduceLROnPlateau`):

```text
Epoch [80/80] | Train Loss: 2.83 (PER: 82.8%) | Val Loss: 3.58 (PER: 82.8%)
pred for every clip: "K A | A AI"   (input-independent common-phoneme collapse)
```

- Tried **both** the tiny model and the full-size model — both stall at ~2.8 loss.
- Root cause isolated: `ReduceLROnPlateau` decays LR from 1e-4 to **~5e-8** by ~epoch 40, freezing the model in a CTC "predict the most common phonemes" local optimum. It is **not** a data/label/CTC/visual bug.
- Confirmed the visual path is healthy: raw logits differ between clips; frames load correctly (decord mean/std sane).

Constant-LR experiments (tiny model; “updates” = optimizer steps):

| LR | 320 updates | 600 | 1000 |
|---|---:|---:|---:|
| 1e-4 | PER 48% | 11.7% | **0.7%** |
| 3e-4 | PER 1.4% | 0.0% | 0.0% |

So the recipe's real problem is **LR collapse + too few updates** (80 epochs × 4 steps = 320), not model capacity.

### Fix applied (minimal, principled)

`config.py`: added `--lr_patience` (default 3) and `--min_lr` (default 1e-5, an LR floor).
`main.py`: `ReduceLROnPlateau(..., patience=args.lr_patience, min_lr=args.min_lr)`.

Defaults preserve the old behavior except the LR can no longer collapse below 1e-5. For a gate/short run, pass `--min_lr 1e-4` to hold the LR.

### Gate result after fix (held LR 1e-4, 736 epochs before a tool timeout)

```text
Epoch [736/800] | Train Loss: 4.34 (PER: 8.3%) | Val Loss: 3.88 (PER: 11.0%) | LR: 1.00e-04
--> Saved new best checkpoint ... (Val PER: 9.0%)
```

Decoding the saved checkpoint on the 8 micro clips: **7/8 decoded exactly correct** (clip 0 wrong). This is genuine memorization → **micro-overfit gate PASS on PER** (train 8.3% / best val 9.0%, well under the ~30% bar).

Caveat: under a *held* LR the CTC loss oscillates late (3–4) even as PER improves; the loss was 0.97 at epoch 300. Interpret the gate by PER/decoded output, not the final loss value. A gently decaying floor (instead of a fully held LR) would settle the loss if a clean curve is wanted.

## 8. Code changes

| File | Change |
|---|---|
| `config.py` | Added `--lr_patience` (default 3) and `--min_lr` (default 1e-5) |
| `main.py` | `ReduceLROnPlateau` now uses `patience=args.lr_patience, min_lr=args.min_lr` |

No dataset/model/CTC logic was changed. Unit suite re-ran green after the edit.

## 9. Artifact locations (E:)

```text
E:\punarvaani_venv\                 # Python 3.11 venv
E:\punarvaani_data\raw\             # extracted zip (raw videos + metadata)
E:\punarvaani_data\raw_split\       # hardlinked speaker-disjoint train/val/test
E:\punarvaani_data\raw_micro\       # 2-speaker micro set
E:\punarvaani_data\cropped\{train,val,test}\
E:\punarvaani_data\cropped_micro\
E:\punarvaani_data\checkpoints\     # stage1_micro_overfit.pth (+ diag checkpoints)
E:\punarvaani_data\logs\            # micro overfit logs
E:\punarvaani_data\preflight_*.json
```

## 10. Stage 2 / adapter

`checkpoints/sarvam_lora/` contains **only README.md** — no `adapter_config.json`, no weights. Stage 2 will run in **MOCK** mode; eval skips CER/WER and inference prints `[MOCK TRANSCRIPTION: ...]`. Adapter validation cannot be exercised until the files are delivered.

## 11. Pending work

- [ ] Run train preflight on `cropped/train`.
- [ ] 3-epoch Stage-1 smoke train on `cropped/train` with `--val_dir cropped/val` (use `--num_workers 0`, `--strict_ctc`).
- [ ] Eval on `cropped/test` (expect Stage-2 MOCK).
- [ ] Single-clip infer (expect phonemes + `[MOCK TRANSCRIPTION: ...]`).
- [ ] Drop the Sarvam-1 LoRA adapter and re-run adapter preflight + eval/infer for real CER/WER.

## 12. Do not claim

No end-to-end WER, no clinical validity, no SOTA. Stage-2 metrics from the home PC (CER 1.4%, WER 4.7% clean) are **text-only** and must not be reported as video WER. The Stage-1 smoke run is a pipeline health check only.

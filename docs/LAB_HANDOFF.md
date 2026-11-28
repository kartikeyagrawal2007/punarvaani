# Punarvaani — Lab PC Handoff (full context for new chat)

> Read this file first and continue from "Current task" below.
> Generated 2026-11-23 from the home-PC chat. Do not assume numbers not listed here — ask if missing.

## 0. Current task

Run a **smoke test on ~1 hour of real Hindi video data + the delivered Sarvam-1 LoRA adapter** on the **college GPU PC** (Windows, Python 3.11 venv, CUDA PyTorch, MediaPipe).

Scope is plumbing + pipeline sanity, NOT a final accuracy claim:
- preflight dataset + adapter
- micro-overfit gate (tiny model, same few clips, loss should collapse)
- short Stage-1 smoke train on ~1h subset (loss finite, checkpoint written)
- eval + single-clip infer (proves end-to-end runs, adapter loads, not MOCK)

Do NOT claim end-to-end WER, clinical validity, or SOTA from this run.

## 1. Repo / machine state

- Repo: `https://github.com/kartikeyagrawal2007/punarvaani.git`
- Branch: `main`, last known tip `e153696`, `main` in sync with `origin/main` on home PC
- Project root (home): `C:\Users\aayus\OneDrive\Desktop\VALLR`
- Lab root (example): `D:\punarvaani` — adjust if different
- Home PC is CPU-only, no CUDA GPU, no MediaPipe. College GPU PC is the training/smoke machine.

Untracked / local-only on home PC at handoff time:
- `docs/GUIDE_PROGRESS_REPORT.md` — guide meeting mind map + Stage-2 metrics (created, NOT committed)
- Empty junk file literally named `0.12.0'` in repo root (0 bytes, untracked — safe to delete, do NOT commit)
- This file `docs/LAB_HANDOFF.md` — commit + push this so lab can pull it

Tracked placeholders that MUST stay:
- `hindi_dataset/README.md`
- `checkpoints/sarvam_lora/README.md`

## 2. What Punarvaani is (for a fresh chat)

Hindi visual speech recognition, two-stage VALLR-inspired:

1. Stage 1: silent mouth video → 55-class Hindi phoneme sequence (CTC, `Models/ML_VALLR.py`)
2. Stage 2: phoneme sequence → Devanagari text via Sarvam-1 + LoRA (`Models/Sarvam.py`)

Why two stages: visemes. `P/B/M` (`प/ब/म`) look identical on lips. Stage 1 predicts sounds, Stage 2 uses Hindi language prior to resolve.

What we used from VALLR (ICCV 2025): two-stage idea + phoneme CTC bottleneck. What we did NOT use: VALLR weights, English/LRS3 data, AV-HuBERT encoder, per-patient adaptation. Stage 1 is our own spatiotemporal ViT trained from scratch. Hindi-specific: 55-class HPI, Hindi G2P with schwa deletion, Sarvam-1 Stage 2.

Key files:
```text
config.py                  # CLI flags + 55-class vocab + WarmupScheduler + crop resolution
main.py                    # train / eval / infer + shared helpers (_amp_context, _ctc_loss, loaders)
inference.py               # transcribe_video() API
Data/dataset.py            # <speaker>/<clip>.mp4 + .txt, speaker-disjoint split, all-frames, CTC T>=U+R
Data/hindi_g2p.py          # Devanagari → HPI phonemes, schwa deletion
Data/metrics.py            # corpus CER/WER
Data/synthetic.py          # synthetic speaker-tree generator for plumbing
Models/ML_VALLR.py         # VideoViTMasked + SpatialAttentionPool + TemporalAdapter + CTC head
Models/Sarvam.py           # HindiPhonemeTranslator, TAGS contract, 7-step adapter validation, MOCK fallback, train_sarvam_lora()
face_cropper.py            # MediaPipe face/mouth cropper (requires mediapipe)
mouth_geometry.py          # MediaPipe-free mouth box maths (unit-testable)
scripts/preflight.py       # dataset + adapter + runtime gate before GPU time
scripts/preprocess_lips.py # offline mouth crop, preserves speaker folders
scripts/make_synthetic_dataset.py
tests/                     # 73 tests claimed; test_smoke_e2e.py is RUN_SLOW=1 only
docs/TRAINING_RUNBOOK.md   # clone-to-checkpoint Windows/A5000 runbook + 4 gates
docs/ARCHITECTURE.md       # as-built contracts
TEAM_GUIDE.md              # beginner onboarding, data/LoRA/training roles
```

Prompt contract (exact, do not change):
```text
<S2S>
<PHONEMES>
N A M A S TD E | BH AA R A TD
</PHONEMES>
<TEXT>
```
Tags: `["<S2S>", "<PHONEMES>", "</PHONEMES>", "<TEXT>"]`, each must encode as 1 token. Base model: `sarvamai/sarvam-1`.

## 3. Data + adapter status (user-reported, do not invent more)

Video:
- User says: 2–3 sec clips with Devanagari transcripts, 500+ speakers collected, more collection ongoing, total hours NOT yet counted
- Later: "received 2 hours of data" + "gotten the lora connectors" + "collected few hours of data"
- Treat "2 hours / few hours" as pilot-scale. Enough to validate pipeline + first Stage-1 baseline, NOT a final research set.
- Missing for a proper report: exact clip count, exact hours, clips/speaker, crop success rate, speaker-disjoint manifest

Stage-2 text (separate from video):
- New v2 set: 60k train (80% noisy / 20% clean), 4k val, 2k test, zero target overlap, high-quality filters (6–22 words, 30–150 chars, pure Devanagari, unique-word ratio ≥0.6)
- Built from 200k source rows → 159,757 unique targets → 104,299 passed filters
- Old 200k set had dups + 23% val overlap — superseded, do not use

Drop points (gitignored except READMEs):
```text
hindi_dataset/              # <speaker>/<clip>.mp4 + <clip>.txt, UTF-8 Devanagari, ≥2 speakers
checkpoints/sarvam_lora/    # adapter_config.json + adapter_model.safetensors/.bin + tokenizer files
sarvam_data/                # Stage-2 JSONL only if retraining Stage 2 (not needed for smoke)
checkpoints/*.pth           # Stage-1 checkpoints (e.g. punarvaani_stage1.pth) — never commit
```

`.gitignore` verified:
- `hindi_dataset/*` ignored except `hindi_dataset/README.md`
- `checkpoints/*` ignored except `checkpoints/sarvam_lora/` dir, and inside it `*` ignored except `README.md`
- `sarvam_data/` ignored, `*.mp4/*.safetensors/*.bin/*.pth/*.jsonl` ignored
- Only `checkpoints/sarvam_lora/README.md` + `hindi_dataset/README.md` are tracked in those paths

Adapter folder must end up as:
```text
checkpoints\sarvam_lora\
├── README.md
├── adapter_config.json
├── adapter_model.safetensors   (or .bin)
├── tokenizer_config.json
├── special_tokens_map.json
└── tokenizer.json / added_tokens.json (if provided)
```
Do NOT copy optimizer/scheduler/trainer_state or whole `checkpoint-15000/` folders.

## 4. Implementation status

- Stage 1: code ready, NO training run yet (home chat confirmed "Code ready, no training")
- Stage 2: LoRA adapter trained, text-only eval done (see §5)
- Tests: `python -m compileall` over Data/Models/scripts/tests + core .py passed on home PC. Full `python -m unittest discover -s tests -v` did NOT complete on home PC (timed out / missing torch+deps). Do NOT claim "all tests pass" until lab run.
- Preflight / dataset / G2P / CTC / Sarvam loader logic read and coherent, but runtime not exercised on home PC.

## 5. Stage-2 metrics (text-only, NOT video)

Adapter: checkpoint 15000, LoRA r=16, alpha=32, tag embeddings saved with adapter. Trained on 60k v2.

| Split | Exact | CER | WER |
|---|---|---:|---:|
| Val clean | 58.10% | 1.49% | 4.76% |
| Test clean | 57.70% | 1.42% | 4.71% |
| Val noisy | 33.20% | 3.94% | 9.95% |
| Test noisy | 31.00% | 3.91% | 9.94% |

Read as:
- Clean = perfect G2P phonemes → text. Noisy = same targets with synthetic viseme swaps/drops/dups (proxy for Stage-1 errors, NOT real Stage-1 output).
- Exact is strict (one char/word fails sentence). CER/WER are the summary.
- Val≈test → generalizes, not just fitting val. Clean→noisy gap ~+5.2 WER points.
- Base Sarvam-1 without adapter: exact 0%, CER 174%, WER 198% — NOT a fair base-model benchmark (tags unseen, random embeddings). Only shows adapter is necessary for this prompt format.

Do NOT report 4.71% as "video WER". Guide report explicitly warns against this.

## 6. Lab setup (college GPU PC)

```powershell
git clone https://github.com/kartikeyagrawal2007/punarvaani.git
cd punarvaani
# if repo already cloned: git pull origin main ; git log --oneline -5 ; git status

python --version   # expect 3.11.x
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt   # pins mediapipe==0.10.14; a bare `pip install mediapipe` gets a version without mp.solutions
# NOTE: requirements.txt contains `triton` (Linux/CUDA-only). On Windows if install fails on that line, delete the `triton` line and rerun.
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "import mediapipe; print(mediapipe.__version__)"
python -m unittest discover -s tests -v   # must end OK (1 skipped slow E2E is normal)
```

Then drop data + adapter (see §3 paths). Do NOT overwrite the two README.md placeholders.

## 7. Smoke test on ~1h (lab GPU)

Assumes repo at `D:\punarvaani`. Replace `$DATA` / speaker names with real ones.

### 7.1 Make ~1h subset (keep full data intact)

```powershell
$src = "D:\punarvaani\hindi_dataset"
$dst = "D:\punarvaani\hindi_dataset_smoke_1h"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
$speakers = @("spk01","spk02","spk03")  # choose enough for ~1h
foreach ($s in $speakers) { robocopy "$src\$s" "$dst\$s" /E }
"MP4: $((Get-ChildItem $dst -Recurse -Filter *.mp4).Count)"
"TXT: $((Get-ChildItem $dst -Recurse -Filter *.txt).Count)"
"Speakers: $((Get-ChildItem $dst -Directory).Count)"
```
Need: MP4==TXT, ≥2 speakers. ~1h ≈ 1200 clips @3s (or ~1800 @2s).

### 7.2 Crop (only if clips are raw faces)

If already mouth-cropped → skip, use `--crop_mode none`.
If raw → preprocess:
```powershell
python scripts/preprocess_lips.py --input_dir D:\punarvaani\hindi_dataset_smoke_1h --output_dir D:\punarvaani\hindi_dataset_smoke_1h_cropped --max_frames 96 --frame_size 224
```
Check success rate + `preprocessing_report.json`, spot-check crops. Then set `$DATA` to the cropped folder.

Crops are stable by default (`--crop_method stable`): the whole clip is read first, MediaPipe landmarks are mapped to full-frame pixels, the crop size comes from the distance between the eyes (it follows camera zoom but never the lips) and the mouth centre, size and head roll are smoothed over 9 frames. The old per-frame crop (`--crop_method legacy`) sized the box from the lip height, so it zoomed out whenever the mouth opened, and smoothed in a coordinate system that moved every frame; crops made with it (for example `E:\punarvaani_data\full_cropped`, 2026-11) should be redone. `--workers N` crops N clips in parallel (default: half the CPU cores, at most 8; each worker loads the models once). Add `--skip_existing` to continue an interrupted run; clips are written under a temporary name and renamed when finished, so a half-written clip is never kept.

Unsure? Open a clip: tight 224x224 mouth ROI = cropped; full face/frame = raw.

### 7.3 Preflight

```powershell
$DATA = "D:\punarvaani\hindi_dataset_smoke_1h_cropped"  # or _smoke_1h if already cropped
$ADAPTER = "D:\punarvaani\checkpoints\sarvam_lora"
python scripts/preflight.py --data_dir $DATA --sarvam_adapter_path $ADAPTER --crop_mode none --output D:\punarvaani\preflight_smoke_1h.json
```
Pass = overall pass, `adapter: compatible=True`, no missing pairs, no non-Devanagari errors, ≥2 speakers, valid CTC, non-zero duration.
If layout is `train/val/test` splits, preflight each split (`--min_speakers 1` for single-speaker val/test) and train with `--data_dir ...\train --val_dir ...\val --val_split 0`.

### 7.4 Micro-overfit gate (fast sanity)

Copy 2 small speaker folders to `D:\punarvaani\hindi_dataset_micro`, then:
```powershell
python main.py --mode train --data_dir D:\punarvaani\hindi_dataset_micro --val_dir D:\punarvaani\hindi_dataset_micro --save_model_path D:\punarvaani\checkpoints\stage1_micro_overfit.pth --val_split 0 --crop_mode none --max_frames 48 --batch_size 2 --eval_batch_size 2 --epochs 300 --warmup_steps 20 --min_lr 1e-4 --num_workers 0 --temporal_upsample_factor 2 --vit_embed_dim 128 --vit_depth 2 --vit_heads 4 --spatial_pool_tokens 4x4 --device cuda
```
Pass = PER well under ~30% and most clips decode exactly (2026-11-23 lab run: 7/8 exact, best val PER 9.0%). Judge by PER and decoded output: with the LR held, the CTC loss can oscillate late even while PER improves. If it fails, STOP and debug data/transcripts/CTC/labels before the bigger run.

Why `--min_lr 1e-4` and 300 epochs: with 8 clips and batch 2 there are only 4 updates per epoch. The old 80-epoch recipe gave 320 updates, and the default plateau scheduler cut the LR to ~5e-8, so the model froze on the most common phonemes. Holding the LR and giving ~1,200 updates fixes it (see `docs/LAB_SMOKE_FINDINGS.md` §7).

### 7.5 Smoke train (~1h, 3 epochs, normal size)

```powershell
python main.py --mode train --language hindi --data_dir $DATA --save_model_path D:\punarvaani\checkpoints\stage1_smoke_1h.pth --sarvam_adapter_path $ADAPTER --val_split 0.2 --crop_mode none --max_frames 96 --batch_size 4 --eval_batch_size 4 --epochs 3 --warmup_steps 100 --strict_ctc --device cuda
```
If OOM → batch 2 or 1. If dedicated held-out speaker folder exists → use `--val_dir <path> --val_split 0`. This is pipeline/health, NOT a final model.

**If training stalls** (train loss flat around 2.9-3.0 and train PER stuck near 77% for several epochs, while val PER swings), the model is stuck predicting the most common phonemes. A lower LR keeps it there; a higher one gets it out (see `docs/LAB_SMOKE_FINDINGS.md`). Start a fresh run with `--lr 3e-4 --min_lr 1e-4 --lr_patience 5` and a new `--save_model_path`, so the LR never drops below 1e-4 while it is still stuck.

**ResNet front end (recommended for training from scratch).** `--frontend resnet` swaps the ViT for the standard lip-reading encoder: a 3D convolution over 5 frames, a ResNet-18 per frame on 88x88 grey mouth crops (random crop and flip while training), then a 6-layer temporal transformer. On the 17-hour corpus the ViT stalled at a train loss of about 2.95 (train PER about 77-80%) with any learning rate; on a 6-clip memorisation check the ResNet front end reached 0% PER in 100 steps where the ViT was still at 26% after 250. Combine it with `--curriculum_epochs 3` (epoch 1 trains on the shortest third of the clips, growing to all clips by epoch 4). It needs far less GPU memory than the ViT, so use batch 16 (drop to 8 if CUDA runs out of memory):
```powershell
python main.py --mode train --language hindi --data_dir E:\punarvaani_data\full_cropped\train --val_dir E:\punarvaani_data\full_cropped\val --val_split 0 --save_model_path E:\punarvaani_data\checkpoints\stage1_resnet.pth --crop_mode none --max_frames 96 --frontend resnet --curriculum_epochs 3 --batch_size 16 --eval_batch_size 16 --epochs 30 --lr 3e-4 --min_lr 1e-4 --lr_patience 5 --warmup_steps 1000 --strict_ctc --device cuda --resume
```
Eval and infer read the architecture from the checkpoint, so their commands stay the same. Resume needs the same `--frontend resnet` flag.

**Getting past the CTC plateau.** Every model first sits at a train loss of about 2.9-3.0 (it outputs the common phonemes) until it learns when each sound happens. On the full 17-hour corpus no run left that stage in 30 epochs, but on 207 clips (`--sample_size 0.01`, same clips as val, 150 epochs, `--min_lr 3e-4 --warmup_steps 50`) the ResNet front end broke through around epoch 60-75 and memorised them (val PER 4.7%, 2026-11-27). `--init_from <that checkpoint>` starts a new full run from those weights, with a fresh optimizer and LR schedule.

**Audio: a label check and a dense training signal.** `scripts/extract_audio.py` writes `<clip>.audio.npy` next to every crop: four 10 ms log-mel frames (80 bands) per video frame, read from the raw clip's sound, so audio frame i lines up with video frame i. Run it once per split (needs ffmpeg on PATH, e.g. with the `punarvaani` conda env active):
```powershell
python scripts\extract_audio.py --input_dir $HOME\Desktop\punarvaani-data\dataset\splits\train --output_dir E:\punarvaani_data\full_cropped_v2\train --skip_existing
```
- `--frontend audio` trains the same Stage-1 model on the clips' sound with the same labels (video is not decoded, so it is fast). Learning phonemes from audio is far easier than from lips: if even this stays at the CTC plateau (train loss ~2.9), the transcripts do not match the clips; if it learns, the labels are fine and lip reading itself is the hard part.
- `--frontend resnet --audio_weight 1.0` adds audio-guided lip training (as in LiRA, Ma et al. 2021): the lip model also predicts each frame's audio features, a dense per-frame target from the clip's own sound, on top of CTC. The training log prints the audio-prediction L1 each epoch.

**Audio teacher (cross-modal distillation).** On the full corpus the audio-only model (same clips, same labels) left the CTC plateau in epoch 2 and reached 33.7% val PER on unseen speakers (2026-11-28), so the labels are learnable; the lip models are the hard part. Audio-feature prediction (`--audio_weight 1.0`) hardly helped (L1 0.86 -> 0.76, about what predicting the average gives). The stronger use of audio is the trained audio model itself: `--teacher_path <audio model>` makes the lip model also match the teacher's per-step phoneme probabilities on each clip's sound (KL, weight `--distill_weight`), which gives it the timing CTC alone cannot find (as in Afouras et al. 2020, "ASR is all you need"). `--init_from <audio model>` on a lip model also copies the audio model's temporal transformer, adapter and phoneme head, so only the lip front end starts from scratch:
```powershell
python main.py --mode train --language hindi --data_dir E:\punarvaani_data\full_cropped_v2\train --val_dir E:\punarvaani_data\full_cropped_v2\val --val_split 0 --save_model_path E:\punarvaani_data\checkpoints\stage1_resnet_teacher.pth --crop_mode none --max_frames 96 --frontend resnet --teacher_path E:\punarvaani_data\checkpoints\stage1_audio.pth --init_from E:\punarvaani_data\checkpoints\stage1_audio.pth --curriculum_epochs 3 --batch_size 16 --eval_batch_size 16 --epochs 30 --lr 3e-4 --min_lr 1e-4 --lr_patience 5 --warmup_steps 1000 --strict_ctc --device cuda --resume
```
The log prints `Teacher KL` each epoch. The teacher is only used in training: eval and infer read the lips alone.

**Resuming after a shutdown.** After every epoch, training writes `<save_model_path minus .pth>_last.pt` (for example `stage1_full_last.pt`) with the model, optimizer, LR schedule, epoch and best val PER. If the PC restarts, run the same command again with `--resume` added: it prints `Resuming from ...` and continues at the next epoch. The epoch that was running when the PC went off is lost. To train longer than planned, rerun with `--resume` and a larger `--epochs`. The best model is still saved to `--save_model_path`, so eval and infer commands don't change.

### 7.6 Eval + infer

```powershell
python main.py --mode eval --data_dir $DATA --save_model_path D:\punarvaani\checkpoints\stage1_smoke_1h.pth --sarvam_adapter_path $ADAPTER --val_split 0.2 --crop_mode none --max_frames 96 --eval_batch_size 4 --device cuda

python main.py --mode infer --infer_video_path "$DATA\<speaker>\<clip>.mp4" --save_model_path D:\punarvaani\checkpoints\stage1_smoke_1h.pth --sarvam_adapter_path $ADAPTER --crop_mode none --max_frames 96 --device cuda
```

## 8. What terminal output means (cheat sheet)

Preflight:
- `dataset: P/V pairs` → P should equal V; lower = missing/bad transcripts
- `N speakers` → need ≥2 for auto speaker-disjoint split
- `invalid CTC X%` → T_ctc<U+R samples (too short / transcript too long); they get dropped
- `adapter: compatible=True` → Stage 2 can load; False/missing → MOCK, CER/WER skipped
- `CUDA=True` → GPU training; False → CPU fallback (slow)
- `Preflight OK` = no blocking issues; fix any `BLOCKING ISSUES` before training

Train epoch line:
```text
Epoch [1/3] | Train Loss: 4.81 (PER: 98.4%, dropped 0) | Val Loss: 4.79 (PER: 97.9%, dropped 0) | LR: 1.00e-04
  --> Saved new best checkpoint ... (Val PER: 97.9%)
```
- Loss finite + trending down = healthy; `dropped` should be 0/small/stable; `Saved new best` = val PER improved + checkpoint written. High PER on 1-epoch smoke is EXPECTED, not a quality signal.

Eval:
- `CTC Loss / PER / Dropped` = Stage-1 decoding ran. Then either `CER/WER` (adapter active) or `[Stage 2] ... MOCK mode ... Reason: ...` (adapter did not load — fix placement/files/base-model/tags).

Infer:
- Steps 1–3 complete + `PREDICTED PHONEMES` + `FINAL HINDI TEXT` (Devanagari) = full pipeline runs. `[MOCK TRANSCRIPTION: ...]` = Stage 2 fallback, not real output. `Using initialized weights` = Stage-1 checkpoint path wrong/missing.

Common failures:
- `Need at least two speakers` → add speaker or use dedicated `--val_dir`
- `crop_mode='mouth' requires MediaPipe` → install mediapipe or use cropped data + `--crop_mode none`
- Non-Devanagari transcript / missing pair → fix `.txt` (UTF-8 Devanagari, no English/digits, same basename)
- Adapter MOCK → check files directly in `sarvam_lora/`, base `sarvamai/sarvam-1`, tag tokenization
- OOM → lower batch/frames; keep tiny-model flags for micro gate only

## 9. Git / contributor note (do not lose this)

Old rewritten commits `8ea2476` + `abe8bbf` had a `Co-authored-by: CommandCodeBot` trailer. Current `main` log is clean (`git log --all --grep='CommandCodeBot'` returns nothing), but GitHub may still show the bot as contributor via cached unreachable objects. Rolling back + pushing does NOT remove that; only GitHub Support purge or repo recreate does. If guide asks, reference those two SHAs.

If `main` ever needs rollback to before Sarvam-fix commit `00ea1a7` (i.e. to `7ecf71f`), home chat left two options (backup branch first, then `reset --hard 7ecf71f` to discard OR `reset --soft 7ecf71f` + recommit without trailer, then force-push). Do NOT do this on lab PC without coordinating — lab should just `pull`, not rewrite history.

Teammate pull after any force-push:
```powershell
git fetch origin
git reset --hard origin/main
```

## 10. Open questions for user (ask, don't assume)

- Exact delivered video layout? (speaker folders? flat? train/val/test? videos+transcripts split?)
- Are smoke clips already mouth-cropped or raw faces?
- Exact clip/speaker/hour counts for the 2h drop?
- Which lab path is the repo? (assumed `D:\punarvaani`)
- Guide meeting emphasis: VALLR background + what we borrowed vs Hindi-specific work + Stage-2 results + gaps/next steps (see local `docs/GUIDE_PROGRESS_REPORT.md` on home PC — NOT yet pushed)

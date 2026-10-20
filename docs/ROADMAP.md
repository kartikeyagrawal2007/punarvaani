# Punarvaani roadmap

The code for every stage is in `main` and tested on CPU with synthetic data. What remains is the real-world part: Hindi data, volunteer recordings, pretrained weights and GPU training. This roadmap turns the 16-week plan into checklists with owners and exit criteria.

**Owners** (from the team split in the README): **D** = Data lead, **V** = Vision model lead, **S** = Speech model lead, **Sys** = System lead.

Tick boxes as you go (`- [x]`) and keep this file up to date in each pull request.

---

## Where we are

| Phase | Weeks | Status |
|---|---|---|
| 0. Code base (all stages, tests, CI, docs) + Lip Lab dashboard | — | ✅ done |
| 1. Setup and weights | 1–2 | ⏳ **next** |
| 2. Hindi data + volunteer recordings | 3–5 | ⬜ |
| 3. Path A on Hindi | 6–7 | ⬜ |
| ✅ Checkpoint: Path A demo | 8 | ⬜ |
| 4. Personalisation + silent-mouthing study | 9–10 | ⬜ |
| 5. Path B (units + vocoder) | 11–12 | ⬜ |
| 6. Hybrid A+B | 13 | ⬜ |
| 7. Real-time phone app | 14 | ⬜ |
| 8. Evaluation, user study, paper | 15–16 | ⬜ |

---

## Phase 1: Setup and weights (weeks 1–2)

Goal: everything needed for training is downloaded and verified, so later phases fail on research questions rather than on setup.

- [ ] **Sys** Get GPU access: Colab Pro, Kaggle (2×T4, 30 h/week) or a college GPU. Note memory and weekly quota.
- [ ] **All** Clone, `pip install -e ".[dev,vision,audio,llm]"`, run `pytest -q` and `punarvaani demo` on your own machine.
- [ ] **All** Run `punarvaani lab --user <you>`: record 5 takes of the 8 default phrases, train, and note your live test score silent vs spoken. This is the first real-lips number, and the takes are reusable personalisation data.
- [ ] **All** Read `docs/LEARNING_GUIDE.md` §1–§5 and the AV-HuBERT, CTC and LoRA papers (guide §12).
- [ ] **V** Download AV-HuBERT Large (`large_vox_iter5.pt`) from github.com/facebookresearch/av_hubert. Load it with `load_fairseq_checkpoint` and check that the missing/unexpected key report is empty (or only `mask_emb`).
- [ ] **V** Sanity check: run the English checkpoint on a few LRS3-style clips and confirm the outputs are not garbage (validates our re-implementation).
- [ ] **V** Download the MediaPipe `face_landmarker.task` model and run `crop_video_file` on a selfie video. Inspect the crops visually.
- [ ] **S** Download mHuBERT-147 (`utter-project/mHuBERT-147`) and IndicF5 (`ai4bharat/IndicF5`). Synthesise one Hindi sentence with IndicF5 and check that the `IndicF5TTS` wrapper matches the current model card.
- [ ] **S** Get pretrained ECAPA weights (SpeechBrain `spkrec-ecapa-voxceleb`) and write a small weight-mapping function, or train ECAPA on VoxCeleb + Hindi speakers.
- [ ] **D** Get a Hindi ASR checkpoint with word timestamps (AI4Bharat IndicWhisper / IndicConformer) and test `HFTranscriber` on a 1-minute news clip.
- [ ] **Sys** Set up experiment tracking (a shared sheet or W&B) and a private Hugging Face Hub repo for checkpoints.

**Exit criteria:** AV-HuBERT loads cleanly; crops look right; IndicF5 speaks Hindi in a reference voice; ASR gives word timestamps.

---

## Phase 2: Hindi data and recordings (weeks 3–5)

Goal: 50–100 hours of clean Hindi lip-reading data, plus silent and spoken recordings from 5–10 volunteers.

**Mined data (D, V)**
- [ ] **D** Make `sources.txt`: Hindi news, lectures and interviews with frontal talking heads (`<url> <speaker_id>` per line). Check each source's terms of use and keep a record.
- [ ] **D** Run `punarvaani mine` on a pilot of ~20 videos. Read `filter_report.json`: if one reason drops most clips (for example `out_of_sync`), fix the source list, not the thresholds.
- [ ] **V** Get pretrained SyncNet (`syncnet_v2`) weights and plug them into `MiningPipeline(sync_scorer=...)`.
- [ ] **D** Scale up to 50–100 h. Keep a per-channel hours table to avoid one anchor dominating.
- [ ] **D** Manually check 100 random clips: transcript correct, lips visible, in sync. Target ≥ 90% good.
- [ ] **D** Freeze splits (`train/val/test.jsonl`, speaker-disjoint) and give the dataset a version number.

**Volunteers (D, Sys)**
- [ ] **Sys** Adapt `punarvaani/recording/CONSENT_TEMPLATE.md` with the college ethics committee; get approval.
- [ ] **D** Build each volunteer's prompt list: the phrase bank (`configs/phrases_hi.txt`) + ~100 balanced sentences (`select_balanced_sentences`).
- [ ] **D** Record 5–10 volunteers, ~30 min each, spoken then silent (`punarvaani.recording.recorder`).
- [ ] **D** Crop and write per-volunteer manifests (`data/users/<id>/train|val|test.jsonl`, with `mode: silent|spoken`).

**Text (S)**
- [ ] **S** Collect Hindi text (IndicCorp / Wikipedia + transcripts) for the character LM and the lexicon corrector.

**Exit criteria:** ≥ 50 h of filtered clips with a filter report; ≥ 5 volunteers recorded with consent; LM corpus ready.

---

## Phase 3: Path A on Hindi (weeks 6–7)

- [ ] **V** Train `configs/hindi_adapt.yaml` (start with a 10 h subset to tune batch size and learning rate, then the full set).
- [ ] **V** Ablation: frozen English encoder vs `hindi` LoRA vs larger rank. Compare with `punarvaani.eval.evaluate.compare` (paired bootstrap).
- [ ] **S** Fit the character LM; tune `lm_weight` and `length_bonus` for beam search on the validation set.
- [ ] **Sys** Evaluate greedy vs beam vs beam+LM: CER, WER and viseme error rate (VER) on the test set.
- [ ] **S** Rebuild one test voice (`punarvaani rebuild-voice`) from a volunteer's old voice notes; IndicF5 speaks with that reference.
- [ ] **Sys** Wire `ClaudeCorrector(fallback=LexiconCorrector(...))` into `PathAPipeline`; measure WER with and without correction.

**Exit criteria:** a Hindi CER/WER baseline with confidence intervals; LLM correction measured; one rebuilt voice.

## ✅ Checkpoint (week 8): Path A demo

- [ ] **Sys** Live demo on a laptop webcam (`python -m punarvaani.app.realtime ...`): phrase mode + open vocabulary + cloned voice.
- [ ] **All** Record a 2-minute demo video. This is the safe result even if later phases slip.
- [ ] **Sys** Report phrase-mode top-1 accuracy and rejection rate on the volunteers' phrase recordings.

---

## Phase 4: Personalisation and the silent-mouthing study (weeks 9–10)

- [ ] **V** For each volunteer, measure base model CER on spoken vs silent takes: **the silent-mouthing gap** (research question 2).
- [ ] **V** Train a per-user adapter with `configs/personalize.yaml` on silent takes; report before/after CER.
- [ ] **V** Ablation on real data: adapter placement (attention only / + projections / + BatchNorm), the same as the synthetic experiment in `scripts/ablation_personalization.py`. Does the synthetic finding hold?
- [ ] **V** Ablation: how many minutes of user data are needed (2 / 5 / 10 / 15 min)?
- [ ] *(stretch)* **V** Use spoken takes to supervise silent ones (align by DTW on lip landmarks).

**Exit criteria:** a table of silent vs spoken CER, before and after personalisation, with a minutes-of-data curve.

---

## Phase 5: Path B (weeks 11–12)

- [ ] **S** Extract mHuBERT-147 features for the training audio; fit k-means (K=1000) with `punarvaani.models.path_b.units.KMeans`; add `units` to the manifests.
- [ ] **S** Compute ECAPA embeddings per speaker and YIN pitch targets; add them to the manifests.
- [ ] **S** Pretrain the vocoder (`configs/vocoder_pretrain.yaml`). This is the longest GPU job, so start it early.
- [ ] **V** Train lips → units with `configs/path_b_units.yaml`, starting from Path A's `merged.pt` encoder.
- [ ] **S** Fine-tune the vocoder on one volunteer's cleaned voice notes (`configs/vocoder_finetune.yaml`).
- [ ] **Sys** Evaluate: ASR-WER of the generated speech, speaker similarity (ECAPA cosine), MOS pilot (5 listeners).

**Exit criteria:** lips → speech in a volunteer's voice, with WER, similarity and MOS-pilot numbers.

## Phase 6: Hybrid A+B (week 13)

- [ ] **S** Train the unit ASR (`python -m punarvaani.training.train_unit_asr`) on audio units + transcripts.
- [ ] **Sys** Run `guide_units` with Path A's corrected text; tune `steps`, `step_size` and `trust`.
- [ ] **Sys** Compare A vs B vs hybrid on WER, MOS and latency.

## Phase 7: Real-time phone app (week 14)

- [ ] **Sys** Phone client (a web page or Android app) streaming camera frames over a WebSocket to `LiveSession.push_frame`, with audio played back on the phone.
- [ ] **Sys** Measure end-to-end latency (p50/p90) per stage; target < 1 s. Pre-render phrase audio.
- [ ] *(stretch)* **V** int8 quantisation / ONNX export of the encoder for on-device inference.

## Phase 8: Evaluation, user study, paper (weeks 15–16)

- [ ] **Sys** Final test-set numbers: CER/WER/VER, phrase top-1 and rejection, latency, with all ablations and paired bootstrap.
- [ ] **S** MOS listening test with 20+ listeners; blind "is this X?" test with family and friends; EER.
- [ ] **All** Paper: dataset (Hindi AV data engine), silent-mouthing gap, personalisation, voice from scraps, LLM correction, A vs B vs hybrid.
- [ ] **All** Release: dataset card (no faces or voices published without consent), code, demo video.
- [ ] *(later)* Approach an ENT or oncology department for an ethics-approved patient validation study.

---

## Risks and fallbacks

| Risk | Early warning | Fallback |
|---|---|---|
| Open-vocabulary WER stays high | Phase 3 CER > 50% | Lean on phrase mode + personalisation; report VER to show the errors are look-alikes |
| Not enough clean Hindi data | Phase 2 filter report keeps < 30% of clips | More lecture sources; lower the target to 30 h and rely more on volunteers |
| AV-HuBERT weights don't load cleanly | Phase 1 key report not empty | Fix the key mapping in `convert_fairseq_state_dict` before anything else |
| GPU quota runs out | Training slower than planned | Train the `small` preset; fewer decoder layers; shorter clips |
| Voice notes too noisy | Quality report SNR < 10 dB | DeepFilterNet + stricter `min_quality`; ask the family for more clips |
| Ethics approval delayed | No approval by week 4 | Record only team members until approved |
| Path B not intelligible | Phase 5 WER far above Path A | Keep Path A as the main system; the hybrid becomes the contribution |

## Definition of done

- A Hindi lip-reading model with a reported CER/WER and confidence intervals
- A measured silent-vs-spoken gap and personalisation gain on real volunteers
- A live demo speaking in a volunteer's rebuilt voice in under 1 second
- A paper draft and a demo video

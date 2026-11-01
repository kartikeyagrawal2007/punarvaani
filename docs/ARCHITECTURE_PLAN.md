# Punarvaani: architecture plan, VALLR core (DRAFT for team review)

> **Status:** draft v2, 1 Nov 2026. This replaces the earlier multi-encoder plan.
> **Decision:** **VALLR is the core.** The repo's `main` is now a VALLR-based codebase (`Models/ML_VALLR.py`, `Models/Sarvam.py`, `Data/hindi_g2p.py`, `face_cropper.py`).
> This plan says what to keep, what to add from the literature review (44 papers), what to compare against, and what is postponed.

---

## 1. Where we are (what's in `main`)

| Part | File | What it does |
|---|---|---|
| Lip cropping | `face_cropper.py`, `scripts/preprocess_lips.py` | MediaPipe face mesh, EMA-smoothed 224×224 mouth crops |
| Hindi G2P | `Data/hindi_g2p.py`, `config.py` | 55-class Hindi Phonetic Inventory (HPI), schwa deletion |
| Stage 1 | `Models/ML_VALLR.py` | Spatiotemporal ViT (tubelets) → spatial attention pool → temporal adapter → phoneme CTC |
| Stage 2 | `Models/Sarvam.py` | Sarvam-1 (2B) + LoRA: `<PHONEMES> … </PHONEMES> <TEXT>` → Devanagari |
| Data loader | `Data/dataset.py` | Video + transcript pairs, CTC length check, speaker-split check |
| Tests | `tests/test_punarvaani.py` | 18 unit tests |
| Handover | `TEAM_GUIDE.md` | Teammate 1: dataset; Teammate 2: Sarvam-1 fine-tuning |

---

## 2. Final architecture (VALLR core)

```
DATA ENGINE (offline)
YouTube / DD News / lectures → yt-dlp → ffmpeg audio
 → IndicWhisper (AI4Bharat Vistaar) transcript + WhisperX word timings → 1–12 s sentence clips
 → decord @25 fps → MediaPipe face track (one speaker; drop shot cuts)
 → face align + mouth crop 224×224 (face_cropper.py)
 → SyncNet (joonson/syncnet_python) on crop + audio → drop out-of-sync / B-roll / dubbed clips
 → labels: Devanagari text + 55-HPI phonemes (hindi_g2p.py) + visemes
 → filter report → speaker-disjoint train/val/test → manifest
 + Phase 2: volunteers (later patients): the same sentences spoken AND silently mouthed

STAGE 1  video → Hindi phonetic streams                           (ML_VALLR, extended)
 mouth crops (all frames at 25 fps; see §3, point 1)
 → Spatiotemporal ViT → spatial attention pool → temporal adapter
   ├─ phoneme CTC head (55 HPI)          ← existing, primary
   ├─ Devanagari character CTC head      ← NEW: dual decoding (VALLR-Pin)
   └─ viseme CTC head (8 classes)        ← NEW: auxiliary, helps प/ब/म ambiguity
 training aids: time masking + frame-drop augmentation (Ma et al. 2022)
 → CTC beam search → n-best phoneme strings + n-best character strings + confidence

 PER-PATIENT ADAPTATION (10–15 min of the patient's silent mouthing)
   user-dependent padding (Kim et al., ECCV 2022)  vs  patient LoRA  vs  both
   + spoken↔silent alignment loss (paired takes)

STAGE 2  phonetic streams → Devanagari sentence                    (Sarvam.py, extended)
 Sarvam-1 (2B) + LoRA
 prompt: <PHONEMES> n-best </PHONEMES> <CHARS> n-best </CHARS> (+ optional conversation context) <TEXT>
 training data: Hindi text corpus → G2P → synthetic noise (viseme substitutions, drops, insertions),
               later re-estimated from Stage-1's real confusion matrix
 SAFETY NET: phrase mode (200–500 key sentences) + confidence threshold → stay silent when unsure

VOICE OUT (to "restore the voice", not just text)
 text → TTS in the patient's own voice: IndicF5 vs OmniVoice (Apache-2.0; Hindi, Kannada, Marathi, Bengali, Tamil)
 voice rebuild: WhatsApp notes → vocal isolation + diarization (VoiceStudio tool / Demucs + pyannote)
                → denoise (DeepFilterNet) → keep only the patient (ECAPA) → clean reference clip
 safeguard: watermark all generated speech
```

---

## 3. Changes to the current code (review these first)

1. **Use the whole sentence, not 16 frames.** `Data/dataset.py` and `config.py` sample **16 frames evenly from the whole video** (VALLR's original setting).
   - A 3-second sentence has 75 frames at 25 fps and roughly 30–40 phonemes. Squeezing it into 16 frames throws away most of the lip movement, and CTC can't output more labels than time steps.
   - **Proposal:** feed all frames at 25 fps, with variable length plus padding/masks; or use overlapping 16-frame windows with a stride and concatenate the features before CTC.
   - This is the most important fix before training on sentences.
2. **Add the character CTC head and the viseme head** to `ML_VALLR` (the VALLR-Pin idea). Devanagari is almost phonetic, so the character stream gives Sarvam-1 a second, independent view.
3. **Extend the Sarvam-1 prompt** to n-best phonemes and n-best characters (not just the single best phoneme string), and add the noise generator for synthetic training pairs.
4. **Add per-patient adaptation:** user-dependent padding in the Conv3d tubelet layer, plus patient LoRA in the ViT blocks.
5. **Add confidence rejection + phrase mode** before anything is spoken (LipVoicer lesson: a wrong guess becomes fluent wrong speech).
6. **Data engine:** add IndicWhisper/WhisperX labels and a SyncNet filter around `preprocess_lips.py`, plus a filter report.
7. **Voice out:** connect Stage 2 text to IndicF5/OmniVoice in the patient's rebuilt voice. Today `main` stops at text.

---

## 4. One comparison: AV-HuBERT encoder

- **What:** the same pipeline (same heads, same Sarvam-1), with the Stage-1 ViT replaced by **AV-HuBERT** (Meta; pretrained on 1,759 h of lip video).
- **Why only as a comparison:**
  - VALLR's gain comes mainly from **phonemes + LLM**, not from its video encoder.
  - AV-HuBERT is the strongest pretrained lip encoder available, so it's the natural fallback if VALLR's ViT struggles with our limited Hindi data.
  - It's one line in the results table, not a second system to maintain.
- **Input difference:** AV-HuBERT takes 88×88 grayscale mouth crops, ours are 224×224 RGB. The crop script can save both.

---

## 5. Postponed: HuBERT / mHuBERT and Path B

- **HuBERT (English, audio): dropped.** It only listens to audio, and our patient produces no audio.
- **mHuBERT-147 (audio, includes Hindi): postponed to Path B.**
  - It only matters if we later build **Path B** (lips → speech tokens → vocoder, no text). There, it turns training audio into speech tokens.
  - It isn't needed for the VALLR route (lips → phonemes → text → TTS).
  - If we start Path B: first check that its tokens keep Hindi sound distinctions (unit-purity test).
- **Multilingual-VSR initialisation (mpc001):** only if both the VALLR and AV-HuBERT encoders struggle.

---

## 6. Literature review → design decisions

| Block | Finding → decision |
|---|---|
| Two-stage design | VALLR (18.7% WER on LRS3 with 30 h labelled data); neuroprostheses (Willett, Metzger): phoneme CTC + language model works → **keep phonemes + LLM** |
| Dual decoding | VALLR-Pin (Mandarin): character + phonetic streams fused by the LLM → **add a Devanagari character head** |
| Data engine | Auto-AVSR and Whisper-label VSR: noisy auto-labels work; LRS3: SyncNet filtering is essential → **IndicWhisper/WhisperX + SyncNet**. No novelty claim on the pipeline itself. |
| Homophemes | Ma et al. 2022: time masking; our viseme head |
| Silent gap | Petridis 2018 (up to 8.5% drop), metric-learning paper (2023) → phase 2 silent takes + alignment loss |
| Per-patient | Kim et al. 2022 user-dependent padding vs LoRA |
| Uncertainty | LipVoicer: wrong text becomes fluent wrong speech → confidence rejection + phrase mode |
| Voice | Lip2Wav: speaker-specific modelling helps; Repova 2021: record 300–1,000 balanced sentences before surgery when possible; face-voice paper: consent + watermarking |
| Evaluation | Report **PER, CER, WER**; spoken and silent separately; small vs large vocabulary; Vistaar/Kathbath for the ASR we use; clinical: WHOQOL-BREF, VHI-30, listener tests |
| Claims | None of the 44 papers has Hindi, camera or patient evidence. The dataset, the silent subset and the patient outcome are the novelty, not the model. |

---

## 7. Experiments (in order)

1. **Frames:** 16 sampled frames vs all frames at 25 fps (PER on sentence clips).
2. **Heads:** phoneme only vs + character (VALLR-Pin) vs + viseme.
3. **Encoder:** VALLR-ViT vs AV-HuBERT (same heads, same Sarvam-1).
4. **Stage 2:** Sarvam-1 with single-best vs n-best input; synthetic noise vs real-confusion noise; vs a lexicon/n-gram fallback.
5. **Silent gap:** spoken→silent PER before/after per-patient adaptation ± alignment loss.
6. **Per-patient:** padding vs LoRA vs both, at 2 / 5 / 10 / 15 minutes of patient data.
7. **Voice:** IndicF5 vs OmniVoice (intelligibility via IndicWhisper WER, speaker similarity, MOS).

---

## 8. Repos and licences

| Repo / model | Licence | Use |
|---|---|---|
| MarshallT-99/VALLR (+ VALLR-Pin paper, arXiv 2512.20032) | CC BY-NC 4.0 | Core (already in `main`) |
| sarvamai/sarvam-1 | see model card | Stage 2 |
| facebookresearch/av_hubert | research | Comparison encoder |
| joonson/syncnet_python | MIT | Sync filter |
| m-bain/whisperX, AI4Bharat/vistaar | BSD-2, MIT | Labels + ASR benchmark |
| AI4Bharat/IndicF5, k2-fsa/OmniVoice | —, Apache-2.0 | Voice out |
| debpalash/VoiceStudio | AGPL-3.0 | External tool only (local API); never copy its code into ours |
| DeepFilterNet, Demucs, pyannote-audio, SpeechBrain ECAPA | MIT / MIT / MIT / Apache | Voice rebuild |
| utter-project mHuBERT-147 | CC BY-NC-SA | Postponed (Path B) |

Several of these are non-commercial. That's fine for research and the ICMR proof of concept; flag it before any product.

---

## 9. Review checklist for the team

- [ ] Agree to switch from 16 sampled frames to all frames (or sliding windows) for sentences?
- [ ] Add the character CTC head + viseme head (VALLR-Pin style)?
- [ ] Sarvam-1 prompt with n-best phonemes + characters?
- [ ] Per-patient adaptation: padding vs LoRA vs both?
- [ ] AV-HuBERT as the single comparison encoder?
- [ ] mHuBERT and Path B postponed: agree?
- [ ] Voice out: IndicF5 vs OmniVoice, plus watermarking?
- [ ] Data engine: IndicWhisper/WhisperX + SyncNet + filter report?

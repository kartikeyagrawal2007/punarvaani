# Deep dive: speaking again in your own voice (Hindi lip-to-speech for voice-loss patients)

## Context
From the shortlist (scam-call detector, lip-to-speech, sand mining), the user picked lip-to-speech. This file covers four things: what already exists, how feasible the project is, an open-ended problem statement, and the end-to-end architecture and workflow for a 4–6 month student project.

---

## 1. What exists today (research as of Oct 2026)
**English lip-to-speech is an active research area with open code:**
| Work | Approach | Code |
|---|---|---|
| Lip2Wav (CVPR 2020) | One speaker, about 20 h of video per person | github.com/Rudrabha/Lip2Wav |
| Intelligible L2S with Speech Units (Interspeech 2023) | AV-HuBERT, then a mel spectrogram plus HuBERT units, then a multi-input vocoder. LRS3 WER 29.8% | github.com/choijeongsoo/lip2speech-unit |
| LipVoicer (ICLR 2024) | Lip-reading to text, then a diffusion model on the mel spectrogram with ASR guidance. LRS3 WER 33.2% | github.com/yochaiye/LipVoicer |
| RobustL2S, ReVISE, DiVISe, NaturalL2S, LightL2S (2023–25) | AV-HuBERT units with unit-HiFiGAN, DDSP, or low-complexity models | partly open |
| Zero-shot personalized L2S (ICASSP 2023) | Voice taken from a face image when no audio exists | paper only |
| Lipper (AAAI 2019) | Includes a Hindi demo, but speaker-dependent | old |

**Clinical side:** Voice banking for laryngectomy patients exists, but uptake is poor. In a Czech study only 18 of 30 patients recorded enough audio, and only 7 used it. Silent-speech research for laryngectomy uses EMG sensors or text output. A Weill Cornell trial (NCT05479643) has not started.

**What I could not find (the gap):** A system that combines all of the following:
- **Hindi or Indic** lip-to-speech (there is no large Hindi lip-reading dataset; vVISWa is the largest found, and it is mostly isolated words)
- **few-shot personalization**, using a few minutes of the patient's video
- the patient's **original voice rebuilt from noisy archival clips**, such as WhatsApp voice notes and family videos
- **real-time, on-phone** use, with only a camera and no sensors
- handling **silent mouthing**, which differs from normal speaking. Every major dataset (LRS3, VoxCeleb) contains people speaking aloud.

India has one of the highest burdens of oral and laryngeal cancer in the world, which makes this a strong real-world motivation.

---

## 2. Open-ended problem statement
> *How can a person who has lost their voice speak again, naturally, in their own language and in their own voice, using only a smartphone camera, a few minutes of their silent mouthing, and whatever old voice recordings their family has?*

**Research questions (each can be a paper section):**
1. **Cross-lingual transfer:** How far can an English-pretrained visual speech encoder (AV-HuBERT) be adapted to Hindi with about 50–100 hours of automatically mined Hindi video?
2. **Silent-mouthing domain gap:** Lips move differently when no sound is produced. How much does accuracy drop, and does a small amount of silent-mouthing data plus LoRA recover it?
3. **Voice from scraps:** How well can a speaker identity be rebuilt from 1–3 minutes of noisy, compressed WhatsApp audio? Measure it with speaker-similarity scores and with family members' recognition.
4. **Context helps reading lips:** Can an LLM that knows the user's context (hospital, home, daily phrases) fix lip-reading errors? Many sounds look the same on the lips, e.g. "pa", "ba" and "ma".
5. **Real time:** Can the pipeline run in under about 1 second on a mid-range phone, or on a phone plus a laptop?

---

## 3. Feasibility (honest)
| Aspect | Rating | Notes |
|---|---|---|
| Base models | ✅ High | AV-HuBERT, lip2speech-unit and LipVoicer are open. Multilingual HuBERT units (mHuBERT-147) and Indic zero-shot TTS (IndicF5, F5-TTS) exist. |
| Hindi audio-visual data | ⚠️ Medium | You have to build it yourself by mining Hindi news, lectures and YouTube with an LRS-style pipeline. This is a contribution in itself. |
| Compute | ✅/⚠️ | LoRA or partial fine-tuning of AV-HuBERT-Large fits on 1×A100 or 2×T4 (Kaggle, Colab Pro, university). Full pretraining is out of scope. |
| Accuracy | ⚠️ Risk | Even the best English open-vocabulary systems have about 30% WER. **Mitigation:** (a) personalization, since speaker-specific models are far better; (b) the LLM context fix; (c) a **phrase-set mode** with 200–500 daily phrases that is near-certain to work as a fallback. |
| Patient data | ⚠️ Ethics | Develop with healthy volunteers mouthing silently. Approach a hospital (e.g. Tata Memorial or a local ENT/oncology department) for a small validation study with IRB or ethics approval. Do not block the project on this. |
| Demo risk | ✅ Low | The phrase-set mode plus personal voice guarantees a working, emotional demo. |

**Verdict:** This is feasible for 2–4 people in 4–6 months if you scope it as **Hindi + personalization + phrase-set first, open vocabulary second**. Expect a dataset contribution, a solid system paper or workshop paper, and a very strong demo.

---

## 4. Architecture

```
Phone camera (25 fps)
   │
   ▼
[1] Face + lip ROI ── MediaPipe Face Mesh → 88×88 grayscale mouth crop, stabilized
   │
   ▼
[2] Visual speech encoder ── AV-HuBERT-Large (pretrained, English)
        + Hindi adaptation (LoRA / partial fine-tuning on mined Hindi AV data)
        + per-user LoRA on 10–15 minutes of the user's silent mouthing
   │
   ├────────────── Path A (intelligibility) ──────────────┐
   │   [3a] CTC/attention head → Hindi text (n-best)      │
   │   [4a] LLM rescoring with user context               │
   │        (phrase list, location, recent conversation)  │
   │   [5a] Zero-shot Indic TTS (IndicF5 / F5-TTS)        │
   │        conditioned on the user's reference voice     │
   │                                                      │
   └────────────── Path B (naturalness, research) ────────┤
       [3b] Unit head → discrete mHuBERT units            │
       [4b] Unit-HiFiGAN vocoder + speaker embedding      │
            (ECAPA from the user's old voice notes)       │
                                                          ▼
                                        Speaker output in the user's voice

Voice reconstruction (offline, once per user):
  old WhatsApp voice notes / videos → denoise (DeepFilterNet) → speech enhancement
  → reference clip + speaker embedding → (optional) LoRA fine-tune TTS on 2–5 min
```

**Chosen scope: Path A first, then Path B, with the hybrid as a stretch goal.**
- **Path A** (lips → Hindi text → LLM fix → personal TTS) is the "stenographer + newsreader" path. It is reliable and the LLM corrects errors, but it is chunked (1–2 s delay) and flat in expression. It is built first so a working demo exists by week 8.
- **Path B** (lips → sound tokens every 20 ms → vocoder in the user's voice) is the "ventriloquist" path. It is continuous and keeps rhythm and emphasis, but it is less accurate. This is the research contribution.
- **Stretch: hybrid.** Path B generates the sound and Path A's text guides it toward the correct words (LipVoicer-style classifier guidance). It aims to be natural like B and accurate like A.
- The report compares A vs. B vs. hybrid on intelligibility, naturalness, voice similarity and latency.

**What you build that is new:**
1. The Hindi lip-to-speech data pipeline and dataset
2. Silent-mouthing adaptation
3. Context-aware LLM correction for lip reading
4. Voice reconstruction from noisy archival audio
5. An end-to-end real-time system

---

## 5. Workflow (16 weeks)
| Weeks | Milestone |
|---|---|
| 1–2 | Read the 6 key papers. Reproduce the lip2speech-unit or LipVoicer LRS3 inference. Set up the GPU environment. |
| 3–5 | **Data engine:** download Hindi news and lecture videos → face tracking → SyncNet audio-video sync filter → IndicWhisper transcripts → clips. Target 50–100 h. Record 5–10 volunteers mouthing silently (about 30 minutes each, plus the same sentences spoken aloud). |
| 6–7 | **Path A:** Adapt AV-HuBERT to Hindi (LoRA) with a text head. Voice reconstruction from voice notes (denoise → zero-shot IndicF5). |
| 8 | **Path A complete:** LLM context rescoring plus phrase-set mode. End-to-end demo (lips → text → own voice). **Safe checkpoint.** |
| 9–10 | Personalization: per-user LoRA on silent-mouthing data. Measure the silent vs. spoken gap. |
| 11–12 | **Path B:** Unit head (mHuBERT units) + unit-HiFiGAN vocoder conditioned on the user's speaker embedding. |
| 13 | **Stretch, hybrid:** Path A text guides Path B generation (classifier guidance / joint decoding). |
| 14 | Real-time app: phone → laptop or edge GPU → audio back. Path A in chunks, Path B streaming. |
| 15–16 | Evaluation of A vs. B vs. hybrid, user study with volunteers (and patients if approved), paper and demo video. |

## 6. Evaluation metrics
- **Intelligibility:** WER and CER of generated speech, transcribed by IndicWhisper; STOI and ESTOI where reference audio exists
- **Voice match:** ECAPA speaker-similarity cosine; blind A/B tests with friends and family ("is this X?")
- **Naturalness:** MOS listening test (20+ listeners)
- **Phrase-set mode:** top-1 accuracy
- **Latency:** end-to-end ms
- **Ablations:** English-only vs. Hindi-adapted; with vs. without personalization; with vs. without LLM; Path A vs. Path B

## 7. Key starting points
- github.com/choijeongsoo/lip2speech-unit (AV-HuBERT → units + mel → vocoder)
- github.com/yochaiye/LipVoicer (text-guided diffusion)
- github.com/facebookresearch/av_hubert, github.com/facebookresearch/speech-resynthesis
- AI4Bharat IndicVoices-R, IndicF5, IndicWhisper; mHuBERT-147; vVISWa (Hindi/Marathi audio-visual, word-level)
- Hegde et al., "Towards Accurate Lip-to-Speech Synthesis in-the-Wild" (ACM MM 2023)

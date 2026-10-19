# Punarvaani (पुनर्वाणी): the voice, returned

*Restoring a patient's own voice from silent lip movements, in Hindi.*

> "Surgery took their voice. Punarvaani gives it back."

---

## 1. The idea

People who lose their voice to throat or oral cancer surgery (laryngectomy), stroke or ALS can still **move their lips**, but no sound comes out. India has one of the highest rates of oral and throat cancer in the world.

**Punarvaani** watches the person's lips through a phone camera and **speaks for them in their own original voice**. It rebuilds that voice from old WhatsApp voice notes and family videos.

## 2. Novelty

**What already exists:**
- English-only lip-to-speech research models: [lip2speech-unit](https://github.com/choijeongsoo/lip2speech-unit), [LipVoicer](https://github.com/yochaiye/LipVoicer), NaturalL2S, RobustL2S
- Voice banking, where patients record their voice before surgery. Few do it; in one study only 7 of 30 patients ended up using it.
- Silent-speech systems that need EMG sensors on the face

**What nobody has combined (our gap):**
1. **Hindi lip-to-speech.** No large Hindi lip-reading dataset exists, so we build one.
2. **Personalisation** to one person from about 10–15 minutes of their video
3. **Voice rebuilt from scraps**: noisy, compressed old voice notes, not studio recordings
4. **Silent mouthing.** All major datasets show people speaking aloud, and lips move differently when silent.
5. **LLM context correction.** "pa", "ba" and "ma" look identical on the lips, so an LLM fixes errors using context.
6. **Real time, phone camera only**, with no special sensors

## 3. Feasibility

| Area | Status | Plan |
|---|---|---|
| Base models | ✅ Open source | AV-HuBERT, lip2speech-unit, LipVoicer, IndicF5, mHuBERT-147 |
| Compute | ✅ Manageable | LoRA or partial fine-tuning on Colab Pro, Kaggle or a university GPU. No pretraining from scratch. |
| Hindi data | ⚠️ We build it | Auto-mine 50–100 h from Hindi news and lecture videos, and record 5–10 volunteers mouthing silently |
| Accuracy | ⚠️ Main risk | Even English SOTA has about 30% word error rate. We counter it with per-user tuning, LLM correction, and a **phrase mode** (200–500 daily phrases) as a guaranteed fallback. |
| Patients | ⚠️ Ethics | Build with healthy volunteers. Approach an ENT or oncology hospital for a small approved validation study later. |

## 4. Architecture

```
 Phone camera (25 fps)
        │
        ▼
 [1] Lip crop ── MediaPipe Face Mesh → stabilised 88×88 mouth region
        │
        ▼
 [2] Visual speech encoder ── AV-HuBERT-Large (Meta, pretrained English)
        + Hindi adaptation (LoRA on our mined Hindi data)
        + per-user LoRA (10–15 min of their silent mouthing)
        │
        ├──── PATH A: "Stenographer + newsreader" (build first, reliable)
        │      lips → Hindi text → LLM fixes errors using context
        │      → IndicF5 text-to-speech cloned to the user's voice
        │
        └──── PATH B: "Ventriloquist" (research contribution)
               lips → speech sound tokens every 20 ms (mHuBERT units)
               → unit-HiFiGAN vocoder conditioned on the user's voice
               (keeps rhythm, pauses, emphasis; no text in between)

 STRETCH: Hybrid. Path B makes the sound, Path A's text keeps the words correct.

 Voice rebuilding (once per user, offline):
   old voice notes / videos → denoise (DeepFilterNet)
   → speaker embedding (ECAPA) + reference clip → used by Path A and Path B
```

**Path A** reads the lips into words, then speaks the words. It is accurate, but a bit robotic and delayed.
**Path B** turns lip movement directly into sound. It is natural and continuous, but harder to make accurate.

## 5. What we build on

| Component | Base | Link |
|---|---|---|
| Lip encoder | AV-HuBERT (Meta) | https://github.com/facebookresearch/av_hubert |
| Lip → speech units + vocoder | Intelligible L2S with Speech Units (Interspeech 2023) | https://github.com/choijeongsoo/lip2speech-unit |
| Text-guided lip → speech | LipVoicer (ICLR 2024) | https://github.com/yochaiye/LipVoicer |
| Unit vocoder | speech-resynthesis (Meta) | https://github.com/facebookresearch/speech-resynthesis |
| Hindi voice cloning TTS | IndicF5 (AI4Bharat) | https://huggingface.co/ai4bharat |
| Hindi transcripts | IndicWhisper / IndicConformer (AI4Bharat) | — |
| Multilingual speech units | mHuBERT-147 | — |
| Lip crop | MediaPipe Face Mesh | — |
| Denoising | DeepFilterNet | — |

## 6. Workflow (16 weeks)

| Weeks | Milestone |
|---|---|
| 1–2 | Read the key papers. Get English lip2speech-unit or LipVoicer running. Set up GPUs. |
| 3–5 | **Data engine:** Hindi videos → face tracking → audio-video sync check (SyncNet) → transcripts → clips (50–100 h). Record volunteers mouthing silently. |
| 6–7 | **Path A:** adapt AV-HuBERT to Hindi and train the text head. Rebuild the voice from voice notes. |
| 8 | **✅ Checkpoint:** Path A end-to-end demo with LLM correction and phrase mode |
| 9–10 | Per-user personalisation. Measure the silent vs. spoken gap. |
| 11–12 | **Path B:** unit head plus a vocoder in the user's voice |
| 13 | Stretch: hybrid A + B |
| 14 | Real-time app (phone → laptop GPU → audio) |
| 15–16 | Evaluation, user study, paper, demo video |

## 7. How we measure success

- **Clarity:** word error rate of the generated speech, transcribed by IndicWhisper
- **Voice match:** speaker-similarity score, plus blind tests with family and friends asking "is this X?"
- **Naturalness:** listener ratings (MOS) from 20+ people
- **Phrase mode:** top-1 accuracy
- **Speed:** end-to-end latency, with under 1 second as the target
- **Comparisons:** English-only vs. Hindi-adapted, with vs. without personalisation, with vs. without LLM, Path A vs. B vs. hybrid

## 8. Team split

| Role | Owns |
|---|---|
| Data lead | `data_engine/`, `recording/` |
| Vision model lead | `preprocessing/`, `models/encoder/` |
| Speech model lead | `voice_rebuild/`, `models/path_b/` |
| System lead | `models/path_a/`, `app/`, `eval/` |

---

## Status

The full code base for every stage is implemented and tested on CPU (80+ tests). Each model is proven to learn on a synthetic lip-reading world; the real Hindi runs need GPUs, the pretrained weights listed in `third_party/README.md`, and the mined data.

| Stage | Code | Verified on |
|---|---|---|
| [1] Lip crop (MediaPipe → stabilised mouth crop) | ✅ `preprocessing/lip_crop.py` | synthetic head motion |
| Data engine (ASR clips, shot/face checks, A/V sync, splits) | ✅ `data_engine/` | fake components end to end |
| [2] AV-HuBERT encoder + Hindi/user LoRA | ✅ `models/encoder/` | fairseq key conversion, LoRA merge exactness |
| Path A: CTC/attention, beam + LM, phrase mode | ✅ `models/path_a/` | synthetic: 0% CER in 400 CPU steps |
| LLM correction (Claude) + offline fallback | ✅ `llm_correction.py` | mocked API, viseme confusions fixed offline |
| Per-user personalisation | ✅ `configs/personalize.yaml` | synthetic "silent" user: 92% → 15.5% CER with 64 clips |
| Voice rebuilding (denoise, ECAPA, speaker clustering) | ✅ `voice_rebuild/` | relatives' voices excluded |
| Path B: lips → units/mel/pitch, unit HiFi-GAN | ✅ `models/path_b/` | 100% unit accuracy; vocoder mel L1 1.18 → 0.68 |
| Hybrid A+B (CTC-guided units) | ✅ `models/hybrid/` | CTC 7.0 → 0.1 nats, 10% of frames edited |
| Real-time streaming | ✅ `app/` | end-pointing + chunked streaming tests |
| Evaluation (WER/CER/VER, bootstrap, MOS, EER, latency) | ✅ `eval/` | checked against jiwer |
| **Next:** mine Hindi data, record volunteers, GPU training | ⏳ | — |

**New to the project? Read [`docs/LEARNING_GUIDE.md`](docs/LEARNING_GUIDE.md):** every component, every design decision, the background knowledge, and interview Q&A.

## Repository structure

```
punarvaani/
├── punarvaani/              # the Python package
│   ├── text/                # Hindi normalisation, character vocab, G2P + schwa deletion, visemes
│   ├── preprocessing/       # [1] lip crop, audio features, pitch
│   ├── data_engine/         # Hindi AV mining: clips, sync check, tracking, manifests
│   ├── datasets/            # manifest dataset, dynamic batching, synthetic lip world
│   ├── models/
│   │   ├── encoder/         # [2] AV-HuBERT-style encoder + stackable LoRA
│   │   ├── path_a/          # lips → text → phrase mode / LLM correction → IndicF5
│   │   ├── path_b/          # lips → units (+mel, pitch) → unit HiFi-GAN
│   │   └── hybrid/          # Path A text guides Path B units
│   ├── voice_rebuild/       # voice notes → denoise → ECAPA → speaker clustering → profile
│   ├── training/            # trainer + train_vsr / train_units / train_vocoder
│   ├── eval/                # metrics + evaluation runner
│   ├── app/                 # real-time streaming + webcam demo
│   ├── recording/           # volunteer prompts, recorder, consent template
│   └── cli.py               # `punarvaani ...` entry point
├── configs/                 # synthetic (CPU) and real (GPU) training recipes, phrase bank
├── tests/                   # pytest suite (fast + one learning test)
├── scripts/                 # reproducible experiments
├── docs/                    # project plan, learning guide
├── notebooks/               # Colab / Kaggle experiments
└── third_party/             # where pretrained weights come from
```

## Setup

```bash
git clone https://github.com/kartikeyagrawal2007/punarvaani.git
cd punarvaani
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # core + tests
pip install -e ".[vision,audio,llm]"    # MediaPipe/OpenCV, audio I/O, Claude API
```

## Quick start (CPU, no data needed)

```bash
pytest -q -m "not slow"                       # ~10 s
punarvaani demo                               # trains a tiny lip reader on synthetic lips, runs Path A
python scripts/ablation_personalization.py    # the per-user LoRA experiment
punarvaani train-units --config configs/synthetic_units.yaml
punarvaani train-vocoder --config configs/synthetic_vocoder.yaml
```

## Real pipeline (GPU)

```bash
punarvaani mine --urls sources.txt --out data/hindi_av --face-model face_landmarker.task
punarvaani train-vsr --config configs/hindi_adapt.yaml
punarvaani rebuild-voice notes/*.opus --out profiles/P07 --enroll notes/known.opus --ecapa ecapa.pt
punarvaani train-vsr --config configs/personalize.yaml lora.name=user_P07 data.train_manifest=...
punarvaani evaluate --run outputs/personalize --manifest data/users/P07/test.jsonl --phrases configs/phrases_hi.txt
python -m punarvaani.app.realtime --run outputs/personalize --phrases configs/phrases_hi.txt --face-model face_landmarker.task
```

## Data & privacy

- **Never commit data or model weights.** Videos live on the shared Drive. Checkpoints live on a private Hugging Face Hub repo.
- **Never upload volunteers' face videos or voice recordings anywhere public.** Every volunteer signs a consent form before recording.
- Patient data is only collected under an approved ethics protocol.

## License & third-party

Our own code is released under the MIT License. Some upstream components carry their own licenses. AV-HuBERT in particular is licensed for non-commercial research only, so the project as a whole is for research use.

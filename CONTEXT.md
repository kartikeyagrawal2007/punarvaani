# Punarvaani: full project context (for teammates and their AI agents)

> **How to use this file:** paste or attach it at the start of a chat with your AI assistant (ChatGPT, Claude, Gemini, etc.) and say:
> *"This is the full context of our project. Read it fully, then help me with …"*.
> Last updated: 24 Oct 2026. Repository: https://github.com/kartikeyagrawal2007/punarvaani

---

## 0. Instructions for the AI assistant reading this

- You are helping a 4-student team at **BMS College of Engineering (BMSCE), Bengaluru** build a deep-learning research project and prepare an **ICMR grant proposal**.
- **Be honest about the state of the project.** All results so far are on **simulated (synthetic) data**. Nothing has been trained on real Hindi data or tested with patients yet. Never present synthetic numbers as real results.
- **Don't over-claim novelty.** Read section 5 first. The models are existing ones; the novelty lies in the combination, the data and the clinical outcome.
- **Explain in simple English.** Most of the team is learning these topics, so explain jargon when you use it.
- **Code work in the repo:**
  - Python 3.10+, PyTorch.
  - Run `pytest -q` and `ruff check punarvaani tests && ruff format --check punarvaani tests` (ruff 0.17.1) before committing.
  - Give branches meaningful names (e.g. `feature/...`, `docs/...`). Commit small docs and fixes straight to `main`; big features go in a PR.
  - **Commits must be authored by the team member only. Do not add AI "Co-Authored-By" lines or any AI attribution to commits or PRs.**
- **No patient data, faces or voices go into git.** Media and data folders are git-ignored on purpose.

---

## 1. The project in one paragraph

**Punarvaani (पुनर्वाणी, "the voice, returned")** helps people who lost their voice to throat-cancer surgery (**laryngectomy**: the voice box is removed, but the lips still move) speak again **in their own voice**. A **phone camera** watches their **silent lip movements**, the system works out what they are saying (in **Hindi**, and later other Indian languages), and it speaks in **their original voice**, rebuilt from **old WhatsApp voice notes and family videos**. It is camera-only and non-invasive: no implants, sensors or special hardware.

**Tagline:** *"Surgery took their voice. Punarvaani gives it back."*

### Why it matters (clinical need)
- India has one of the world's highest burdens of oral and throat cancer: head-and-neck cancers are among the most common cancers in Indian men (National Cancer Registry Programme, Mathur et al., *JCO Global Oncology* 2020, doi:10.1200/GO.20.00122).
- Current options are poor:
  - An **electrolarynx** sounds robotic and costs a lot.
  - **Writing or typing** is slow.
  - **Voice banking** (recording your voice before surgery) is rarely used. In a Czech clinical study, 61 patients were offered it and 30 were eligible; **18 recorded enough, and only 7 used it** (Repova et al., *Biomed Pap* 2021, doi:10.5507/bp.2020.016).

---

## 2. Team, guide and roles

| Person | Role in team (how we introduce ourselves) | Presents slides |
|---|---|---|
| **Kartikey Agrawal** (team lead, repo owner) | Architect: thinks from first principles, designs how the parts fit together | 1–4 (title, problem, research question, two paths) and closing |
| **Aayush Chougule** | Researcher, the team's toughest critic: challenges every idea to find its strongest version | 5–6 (architecture with speech tokens, data) |
| **Chetan Bagaria** | The team's "compass": keeps us on the right direction and flags when we go off track; also an implementer | 7–8 (already running, accuracy risk) |
| **Rishikesh B.M.** | Implementer: executes tasks down to the last detail | 9–11 (16-week plan, scaling/ICMR, team & asks) |

- **Guide: Prof. Sunag (BMSCE).** He agreed to be **Co-PI** on the ICMR proposal. He wants the **PI** to be an experienced researcher in this field who already has **2–3 funded projects**, and he has someone in mind (name and institution not yet confirmed).
- **Code ownership (from the README):** Data lead → `data_engine/`, `recording/`; Vision model lead → `preprocessing/`, `models/encoder/`; Speech model lead → `voice_rebuild/`, `models/path_b/`; System lead → `models/path_a/`, `app/`, `eval/`. These leads are not formally assigned to people yet.

---

## 3. How the system works (architecture)

```
 Phone camera (25 fps, silent patient: VIDEO ONLY, no sound)
        │
        ▼
 [1] Lip crop: MediaPipe Face Landmarker (478 face points) → head motion removed → steady 88×88 mouth crop
        │
        ▼
 [2] Lip reader: AV-HuBERT Large (Meta; 24-layer Transformer, pretrained on ~1,759 h of English video)
        + Hindi LoRA (small add-on layers trained on mined Hindi data)
        + patient LoRA (10–15 min of that patient's silent mouthing)
        │
        ├── PATH A, "Stenographer + Newsreader" (reliable, built first)
        │     lips → Hindi text (CTC + attention decoder, beam search + character language model)
        │     → fix look-alike mistakes (प/ब/म look identical on lips) with an LLM (Claude) or an offline lexicon corrector
        │     → phrase mode (200–500 everyday sentences; stays silent when unsure)
        │     → IndicF5 (AI4Bharat) speaks the text in the patient's cloned voice
        │
        └── PATH B, "Ventriloquist" (research contribution)
              lips → speech tokens every 20 ms (+ pitch) → Unit HiFi-GAN vocoder + patient's voice fingerprint → their voice
              (no text in between, so rhythm, pauses and emotion are kept; streams in ~0.5 s chunks)

 HYBRID (stretch): Path A's text guides Path B's tokens so B stays on the right words.

 VOICE REBUILD (once per patient, offline):
   old WhatsApp notes / family videos → denoise (DeepFilterNet or Wiener filter)
   → ECAPA-TDNN speaker embedding ("voice fingerprint") + clustering to keep ONLY the patient's voice
   → clean ~10 s reference clip + embedding, used by IndicF5 (Path A) and the vocoder (Path B)

 DATA ENGINE (training time):
   Hindi news/lecture videos → Whisper/IndicWhisper word timings → face + lip-sync filters (SyncNet)
   → mouth crops → speaker-disjoint train/val/test splits (target 50–100 h) + volunteer recordings (spoken AND silent)
```

### Path B explained simply (speech tokens)
1. **Make an "alphabet of sounds" (once).** Run lots of Hindi speech through **mHuBERT-147** (a speech model trained on 147 languages), cut it into 20 ms slices, and group the slices with **k-means into 1,000 types of sound**. Every 20 ms of speech becomes one number from 0 to 999 (a "speech token").
2. **Training.** Use videos where we see the lips *and* hear the audio. The **audio is converted into tokens**, which are the correct answers. The model **looks only at the lips** and predicts the tokens: **2 tokens per video frame** (video is 40 ms per frame, tokens are 20 ms). It's a 1,000-way classification trained with cross-entropy, plus pitch prediction.
3. **With the patient.** There is only video and no audio at all. The model predicts tokens from the lips, and the **vocoder** turns them into a waveform in the patient's voice using their voice fingerprint.
4. **Why tokens instead of raw audio:** 1 second of audio is 16,000 numbers, which is far too hard to predict from lips. Tokens turn the task into classification, keep content and timing, and drop background noise; the speaker identity is added back by the vocoder.

---

## 4. What is already built (status)

**Repository:** https://github.com/kartikeyagrawal2007/punarvaani. About 6,500 lines of Python. `main` has **82 automated tests** with CI (GitHub Actions + ruff).

| Stage | Code | Verified on (synthetic data) |
|---|---|---|
| Lip crop (MediaPipe → stabilised crop) | `punarvaani/preprocessing/lip_crop.py` | synthetic head motion |
| Data engine (ASR clips, face/sync checks, splits) | `punarvaani/data_engine/` | fake components end to end |
| AV-HuBERT encoder + stackable LoRA | `punarvaani/models/encoder/` | fairseq key conversion, LoRA merge exactness |
| Path A: CTC/attention, beam + LM, phrase mode, LLM correction, IndicF5 wrapper | `punarvaani/models/path_a/` | toy lip reader 0% error in 400 CPU steps |
| Personalisation (per-user LoRA) | `configs/personalize.yaml`, `scripts/ablation_personalization.py` | simulated "silent" user: **92% → 15.5% error** with 64 clips |
| Voice rebuild (denoise, ECAPA, speaker clustering) | `punarvaani/voice_rebuild/` | relatives' voices excluded |
| Path B: lips → units/mel/pitch, unit HiFi-GAN | `punarvaani/models/path_b/` | 100% unit accuracy; vocoder validation mel L1 0.98 → 0.68 |
| Hybrid A+B (CTC-guided units) | `punarvaani/models/hybrid/` | CTC 7.0 → 0.1 nats, 10% of frames edited |
| Real-time streaming | `punarvaani/app/streaming.py`, `realtime.py` | end-pointing + chunked streaming tests |
| Evaluation (WER/CER/VER, bootstrap, MOS, EER, latency) | `punarvaani/eval/` | checked against jiwer |
| Volunteer recording + consent template | `punarvaani/recording/` | — |

**Lip Lab dashboard** (PR #5, branch `feature/lip-lab-dashboard`, **not yet merged into main**; adds 9 tests, 91 in total):
- `punarvaani lab --user <name>` opens a local web dashboard. You mouth phrases at the webcam; it records takes, trains two small readers (DTW template matching and a temporal CNN on lip-landmark features) in about a minute on a laptop, then quizzes you and keeps a live score (accuracy, coverage, silent vs spoken, confusion table).
- It is a **phrase-mode baseline on one person's face only**. It is **not** the AV-HuBERT model and cannot read free sentences.
- Real-face results so far have been **poor**. That is expected for a tiny few-shot baseline, but it hasn't been diagnosed yet (likely causes: takes cut by pauses, too few takes, strict rejection, head turns).
- `punarvaani lab --synthetic` runs it with a cartoon face, no camera needed.

**What is NOT done yet (real-world part):** downloading pretrained weights and checking they load; mining real Hindi data; recording volunteers; GPU training; any patient work; a clinical partner.

### How to run (CPU, no data needed)
```bash
git clone https://github.com/kartikeyagrawal2007/punarvaani.git && cd punarvaani
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ".[vision]" for MediaPipe/OpenCV (needed for the Lip Lab)
pytest -q -m "not slow"            # fast tests
punarvaani demo                    # tiny lip reader on synthetic lips, then Path A
python scripts/ablation_personalization.py
# Lip Lab (on its branch until PR #5 is merged):
git checkout feature/lip-lab-dashboard && pip install -e ".[dev,vision]" && punarvaani lab --user yourname
```
On a Mac, allow camera access for your terminal app (System Settings → Privacy & Security → Camera), then fully quit and reopen the terminal.

### Libraries
- **Core:** PyTorch, NumPy, PyYAML.
- **Optional:**
  - `vision`: MediaPipe, OpenCV.
  - `audio`: torchaudio, soundfile.
  - `llm`: anthropic.
  - `dev`: pytest, ruff, jiwer.
- **Loaded only by some features:** Hugging Face `transformers` (mHuBERT, IndicF5, Whisper), DeepFilterNet, sounddevice, and the `yt-dlp`/`ffmpeg` command-line tools.
- **Pretrained models:** AV-HuBERT, mHuBERT-147, IndicF5, ECAPA (SpeechBrain), SyncNet, Whisper/IndicWhisper, MediaPipe Face Landmarker.

### Key docs in the repo
| File | What |
|---|---|
| `README.md` | Overview, architecture, status |
| `docs/LEARNING_GUIDE.md` | Every component, design decisions, bug stories, interview Q&A |
| `docs/ROADMAP.md` | Phase-by-phase checklist with owners, exit criteria, risks |
| `docs/PREPARATION.md` | Guide-call preparation, slide outline, likely questions, related work, ICMR facts |
| `docs/project_plan.md` | Original deep-dive |
| `punarvaani/app/lab/README.md` | Lip Lab usage (on the PR branch) |

---

## 5. Novelty: the honest version (important)

**Already done elsewhere, so we must NOT claim these:**
- Lip reading and lip-to-speech in English: LipNet 2016, LRS2/LRS3, AV-HuBERT 2022, Lip2Wav 2020, speech-unit lip-to-speech (Choi et al., Interspeech 2023), LipVoicer (ICLR 2024).
- **Silent (unvoiced) lip reading in English:**
  - The silent-vs-spoken gap was measured at up to 8.5% accuracy loss (Petridis et al. 2018, arXiv:1802.06399).
  - That gap has since been reduced (arXiv:2305.14203, 2023).
- Speech from silent articulation using ultrasound + lip video (arXiv:2304.05574, 2023).
- Voice restoration after laryngectomy using **magnets on the lips and tongue** (2018, trained on recordings made before surgery).
- A patient's own voice restored with a **brain implant** (Metzger et al., *Nature* 2023).
- Picking a voice to match a face (zero-shot, arXiv:2305.14359).
- Adapting AV-HuBERT to Hindi on its own would be judged "process innovation".

**What we found no evidence of (our real novelty):**
1. A **camera-only, non-invasive** system that restores a laryngectomy patient's **own voice**.
2. That voice **rebuilt from everyday recordings** (noisy WhatsApp notes and family videos with other speakers), with no recording needed before surgery. A 2026 paper (arXiv:2606.09667) calls missing voice data for laryngectomy patients a "fundamental challenge".
3. In **Hindi / Indian languages.** No sentence-level Hindi lip-reading dataset exists; only small word-level work (vVISWa 2014, Patil et al. 2019).
4. **Validated with real laryngectomy patients.** No video-only system has been tested on patients.

**World-first claim to use:**
> *The first non-invasive, camera-only system that restores a laryngectomy patient's own voice, rebuilt from everyday recordings, in Indian languages, validated with patients.*

**Verdict:** the project is **not novel as machine learning**, but it is **novel as applied/translational research**. The claim only becomes convincing with a **clinical partner and real pilot data**. Our searches were targeted, so a formal prior-art check (Google Scholar, IEEE Xplore, patents) is still needed before claiming "first in the world".

---

## 6. ICMR "First in the World" Challenge 2026 (the grant we're targeting)

From the **official 2026 call PDF** (`1790846907_call_fiw_2026.pdf`, 8 pages):

- **Purpose:** health technologies "never been thought of, tested or tried anywhere in the world"; "high risk, high reward". "Mere increment in the knowledge or process innovation will not be considered."
- **Tracks:**
  - **Proof of concept:** up to ₹1 Cr, max 2 years, "conceptual stage". **Our track.**
  - Prototype: up to ₹4 Cr, max 3 years.
  - Final product/model: up to ₹8 Cr, max 4 years.
  - Funding is linked to deliverables, and the next stage can be requested later.
- **Eligibility:**
  - One PI per team. A team can come from one or several institutes.
  - Scientists or faculty at government medical colleges, INIs, ICMR institutes, universities, recognised R&D labs and NGOs can apply. **Institutes under UGC/AICTE/NMC need no DSIR certificate.**
  - DSIR-registered private institutions and NGOs (NGO DARPAN), and Indian startups/MSMEs/industry, are also eligible.
- **BMSCE is eligible:** it is AICTE-approved, UGC-autonomous since 2008 and permanently affiliated to VTU, so no DSIR certificate is needed.
- **Co-PI:** "Inclusion of at least one CO-PI from PI's institute is mandatory."
  - If the PI is from BMSCE, Sunag sir covers this.
  - If the PI is from another institution, a Co-PI from *that* institution is also needed, sir joins as an additional Co-PI, and the proposal is submitted from the PI's institution.
  - A hospital clinician can be an additional Co-PI.
- **What the call does NOT require:** no minimum publications, degree, age, permanent post or years of service. General ICMR rules (not in this call) expect a regular post with 3+ years before retirement and a cap of 5 ongoing ICMR projects or ₹25 Cr.
- **Track record still counts:**
  - The short CV asks for date of birth, domain expertise, **PubMed articles in the past 10 years, h-index**, fellowships, up to 10 publications or patents (impact factor and author type), experience as investigator, and ongoing ICMR projects.
  - Part B asks for **5 relevant publications/achievements** and **preliminary work, with proof of concept if applicable**.
- **Selection:** a committee of experts. (1) Confirm the proposal is truly first in the world. (2) Shortlist on relevance and impact. (3) Discuss and decide. **No presentation round is mentioned.**
- **After selection only:** institutional undertaking, **IEC (ethics) approval "as applicable"**, bank details. These are not needed at submission.
- **Submission:** only on the **ICMR e-PMS portal** (open from 07/10/2026), after completing the mandatory PI profile. Any incomplete section means **technical rejection**. Queries go only to **po.epms@icmr.gov.in**.
- **Deadline: 30/11/2026, 05:30 PM. Results: 31/12/2026.**
- **Part A (anonymous: no PI or institute names):**
  - title ≤25 words;
  - summary ≤350 words: rationale 50, novelty 50, objectives 80, **methods incl. regulatory strategy 100**, outcome/impact 70;
  - 6 keywords; ≤10 abbreviations;
  - problem statement ≤500 words; rationale ≤250 words with bibliography; gaps ≤50 words;
  - **≤4 objectives** of ≤20 words each;
  - significance 750 words: clinical need, technical solution, applications, *status of your technology incl. regulatory status and own experimental results*, current best practice and competition, superiority;
  - methodology 1000 words: steps, sample size, statistics, and validation (internal, external, prototype demonstration, technology readiness, commercialisation, ethics);
  - GANTT/PERT chart; deliverables ≤200 words; future plans/technology transfer ≤100 words; conflict of interest ≤30 words; IP status ≤30 words;
  - supplementary PDF ≤5 MB; institutional support and clinical coordination.
- **Part B:** preliminary work ≤250 words; team skills; institutional support with the roles of the PI and Co-PIs; lab facilities; detailed budget.
- **Budget:** staff (ICMR salary scales), equipment (quote, justification, mode of disposal), contingency/consumables by year, travel, overheads "as per rules". **Every item needs a justification**, or it is removed or the proposal is not reviewed. GPU/cloud compute is not mentioned, so justify it under equipment or contingency.

### Plan and status
- **Decided:** Sunag sir as Co-PI; an experienced PI (sir's candidate, TBD); Proof of Concept track.
- **Needed:** a **clinical partner** (ENT/head-and-neck surgeon and/or speech-language pathologist). Suggested: Kidwai Memorial Institute of Oncology (Bengaluru), AIISH (Mysuru), NIMHANS. They give patient access, a DHR-registered ethics committee, and PubMed publications for the CV.
- **Target dates:**
  - finalise the PI and clinician by ~10 Nov (go/no-go);
  - small real pilot (volunteer silent videos + real voice notes) by ~18 Nov;
  - draft to sir by ~20 Nov;
  - submit by **25–28 Nov**.
- **Proposed objectives (≤4):**
  1. Hindi audio-visual dataset with spoken and silent takes from volunteers and consenting patients.
  2. Personalised lip reader for silent mouthing.
  3. Rebuild each patient's own voice from everyday recordings and drive speech from the lips.
  4. IEC-approved smartphone pilot with laryngectomy patients.
- **Clinical framing:**
  - Outcome measures: listener-rated intelligibility, speaker similarity, family "is this them?" test, time to communicate, and a validated communication/voice quality-of-life questionnaire.
  - Ethics: written consent, mouth-only crops, no public faces or voices, ICMR AI ethics guidelines 2023, the DPDP Act, and voice-misuse safeguards.
  - Regulation: assistive software as a medical device (SaMD); CDSCO class to be confirmed; the pilot is a non-interventional feasibility study.
- **Backup funding:** ICMR small/ad-hoc grants, BIRAC BIG (up to ₹50 lakh), the ANRF–ICMR MAHA MedTech Mission, IndiaAI–NCG cancer-AI challenges.

---

## 7. Presentation (12-slide deck) and the talk

**Deck:** teal and coral colours with a sound-wave motif. Latest file: `Punarvaani_v4_12slides.pptx`.
1. Title
2. Problem story
3. Research question / gap
4. One camera, two ways to speak (Path A / B)
5. Architecture
6. Data from scratch, voice from scraps
7. Already running (82 tests, 92→15%, 0%, 6/6; simulated data)
8. Accuracy risk and four layers of defence (personalisation, language model, LLM context correction, phrase mode)
9. The road ahead (16 weeks)
10. Scaling up (ICMR, 2-year plan, more languages)
11. Team & asks
12. "It already runs. We built it before we pitched it."

**Known fix needed:** slide 3 lists "Silent lip movements" under "What nobody has built". That is inaccurate (see section 5). It should read *"Silent speech in Indian languages, on real patients"*.

**Talk split:**
- Kartikey: slides 1–4 and the close.
- Aayush: 5–6.
- Chetan: 7–8.
- Rishikesh: 9–11.
- Kartikey closes: "So, sir, will you guide us?" Adapt this now that sir is Co-PI.

---

## 8. Roadmap (16 weeks, see `docs/ROADMAP.md`)

| Phase | Weeks | Content |
|---|---|---|
| 0 | done | Code base, tests, CI, docs |
| 1 | 1–2 | GPU access; download and verify AV-HuBERT, mHuBERT-147, IndicF5, ECAPA, MediaPipe model; Hindi ASR with word timings |
| 2 | 3–5 | Mine 50–100 h of Hindi data; record 5–10 volunteers (spoken + silent) with consent; text for the language model |
| 3 | 6–7 | Path A on Hindi; beam search + LM; LLM correction; rebuild one voice |
| ✅ | 8 | **Checkpoint: Path A demo** (the safe result) |
| 4 | 9–10 | Personalisation + silent-mouthing study |
| 5 | 11–12 | Path B (units + vocoder in a volunteer's voice) |
| 6 | 13 | Hybrid A+B |
| 7 | 14 | Real-time phone app (<1 s latency) |
| 8 | 15–16 | Evaluation, user study, paper, demo video |

**Main risks:**
- Open-vocabulary accuracy is low (even English state of the art is about 30% word error rate). Answer: personalisation, LM/LLM correction and phrase mode.
- Not enough clean Hindi data.
- Noisy voice notes.
- Ethics approval delays.
- Path B may not be intelligible. Answer: Path A is the safe result.

---

## 9. Literature review (in progress)

- **Format:** 32 core papers (8 per person) plus 11 bonus technical papers (2–3 per person). Deadline: 2 days.
- **Method:**
  1. Read the original abstract yourself.
  2. Give the PDF to GPT with the team prompt (a ~400–500-word summary covering problem, method in simple words, experimentation, key results with exact numbers, conclusion, limitations, and what we can derive for Punarvaani).
  3. If anything feels short, ask GPT to explain it in more detail.
  4. Check every number against the paper.
  5. Fill the team Google Doc table: Paper Title | Year & Venue | Link | Experimentation | Key Results | Conclusion | Limitations | What we can derive | Read by. Paste the full summary under "Detailed notes".
- **Kartikey (lip-to-speech, Path B, personalisation):** Lip2Wav (CVPR 2020, arXiv:2005.08209); Video-to-Speech with GANs (arXiv:2104.13332); SVTS (arXiv:2205.02058); L2S in the Wild with Multi-task Learning (arXiv:2302.08841); ★ Intelligible L2S with Speech Units (arXiv:2305.19603); LipVoicer (arXiv:2306.03258); Let There Be Sound (arXiv:2308.15256); Speaker-adaptive Lip Reading (arXiv:2208.04498). *Bonus:* HuBERT (2106.07447), mHuBERT-147 (2406.06371), Speech Resynthesis (2104.00355).
- **Aayush (lip reading and silent speech, the novelty check):** LRS3-TED (1809.00496); ★ AV-HuBERT (2201.02184); VSR for Multiple Languages (2202.13084); Auto-AVSR (2303.14307); VSR with Whisper labels (2309.08535); ★ Normal/Whispered/Silent speech (1802.06399); Normal-vs-silent gap with metric learning (2305.14203); Speech from silent tongue + lip articulation (2304.05574). *Bonus:* CTC (Graves 2006), Hybrid CTC/Attention (Watanabe 2017), Deep AVSR (1809.02108).
- **Rishikesh (foundations, datasets, Indian languages):** LRW (doi:10.1007/978-3-319-54184-6_6); LipNet (1611.01599); LRS2 (1611.05358); Vid2Speech (1701.00495); Digital Voicing of Silent Speech (2010.02960); Zero-shot personalised L2S with face-based voice (2305.14359); IndicVoices-R (2409.05356); ★ Devanagari lip reading (doi:10.14201/ADCAIJ2019831326). *Bonus:* HiFi-GAN (2010.05646), S2ST with Discrete Units (2107.05604), F5-TTS (2410.06885).
- **Chetan (clinical need and patient systems):** ★ Repova 2021 voice banking (doi:10.5507/bp.2020.016); NCRP India cancer statistics 2020 (doi:10.1200/GO.20.00122); GLOBOCAN 2020 (doi:10.3322/caac.21660); SSI for Speech Restoration review (arXiv:2009.02110); Denby 2010 SSI review (doi:10.1016/j.specom.2009.08.002); Real-time silent speech for laryngectomy (2018, ScienceDirect S021446031830069X); Speech neuroprosthesis + avatar (Metzger, Nature 2023, doi:10.1038/s41586-023-06443-4); Speech neuroprosthesis (Willett, Nature 2023, doi:10.1038/s41586-023-06377-x). *Bonus:* LoRA (2106.09685), ECAPA-TDNN (2005.07143).

---

## 10. Open tasks (as of 24 Oct 2026)

- [ ] Sir to confirm the **PI** (name + institution); confirm BMSCE (or the PI's institution) is registered on ICMR e-PMS
- [ ] Find a **clinical partner** by ~10 Nov
- [ ] Finish the **literature review** (2 days) and the formal **prior-art check**
- [ ] Merge **PR #5 (Lip Lab)**; diagnose poor real-face results (try "Hold Space" mode and 5+ takes per phrase)
- [ ] Record a small **real pilot**: volunteer silent-mouthing videos + real WhatsApp-quality voice notes, with numbers
- [ ] Fix deck slide 3 wording
- [ ] Draft ICMR Part A/B in the exact format (word limits above); consider a provisional patent
- [ ] Phase 1 of the roadmap: GPU access, pretrained weights

---

## 11. Glossary

- **Laryngectomy:** surgical removal of the voice box (larynx), usually for throat cancer. The patient can still move their lips and tongue but can't produce voice.
- **Lip reading / VSR (visual speech recognition):** recognising speech from lip video.
- **Lip-to-speech (L2S):** generating audio speech directly from silent lip video.
- **Silent mouthing / unvoiced speech:** moving the lips without making sound. Lips move differently than in normal speech.
- **AV-HuBERT:** Meta's self-supervised audio-visual speech model; our lip-reader backbone.
- **LoRA:** small trainable add-on layers that adapt a big model cheaply (for Hindi and for each patient).
- **CTC:** a training loss that lets a model output text without needing frame-by-frame alignment.
- **Speech tokens / units:** numbers (0–999) for 20 ms slices of speech, made by mHuBERT features + k-means.
- **Vocoder (HiFi-GAN):** turns tokens or spectrograms into an audio waveform.
- **ECAPA-TDNN:** model that makes a speaker embedding ("voice fingerprint").
- **IndicF5:** AI4Bharat's Indian-language text-to-speech with voice cloning.
- **Phrase mode:** recognising one of 200–500 known sentences; stays silent when unsure (the safety net).
- **WER/CER:** word/character error rate (lower is better). **MOS:** listener rating of naturalness (1–5).
- **IEC:** Institutional Ethics Committee. **PI / Co-PI:** Principal / Co-Principal Investigator. **e-PMS:** ICMR's proposal portal.
- **SaMD:** Software as a Medical Device. **CDSCO:** India's medical device regulator.

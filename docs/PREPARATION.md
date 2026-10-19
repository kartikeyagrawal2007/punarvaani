# Preparation: team discussion and call with Sunag sir

**Purpose of the call:** get Sunag sir's support to apply to the **ICMR "First in the World" Challenge Grant (2026)** with Punarvaani, agree on PI/Co-PI, and agree on the next steps to the deadline.

**Call:** tomorrow, 12:30 PM (he travels in the evening, so keep it to ~20 minutes).

This file has six parts:

1. [Team meeting before the call](#1-team-meeting-before-the-call): decide these first
2. [Call flow](#2-call-flow-what-to-say-in-what-order): what to say, in what order, with talking points
3. [Slide deck outline](#3-slide-deck-outline): 12 slides you can turn into a PPT directly
4. [Questions he may ask](#4-questions-he-may-ask-and-answers), the ICMR facts and the proposal timeline
5. [Libraries, models and tools](#5-libraries-models-and-tools): what we use at each stage and what each one lets us do
6. [Related work, the real gap and our research question](#6-related-work-the-real-gap-and-our-research-question): what already exists (including English), why we focus on Indian languages, and the open-ended question

---

## 1. Team meeting before the call

Agree on these **before** talking to sir. Arriving with clear answers is what makes the pitch convincing.

| # | Decide | Suggested answer |
|---|---|---|
| 1 | Who presents on the call | One speaker (Kartikey); others add on their own topic only |
| 2 | Who we ask to be **PI** | Sunag sir |
| 3 | Who could be **Co-PI** (must be from the same institution) | A second faculty member in AI / signal processing / biomedical; list 2 names |
| 4 | Hospital partner | Any ENT / head-and-neck oncology doctor anyone knows; list contacts |
| 5 | Which ICMR track | **Proof of Concept**: up to ₹1 Crore, max 2 years |
| 6 | Team roles | Data lead, Vision model lead, Speech model lead, System lead (README team split) |
| 7 | Who writes which proposal section | See [proposal timeline](#proposal-timeline-to-30-november) |
| 8 | What we show on the call | The GitHub repo, `docs/ROADMAP.md`, and the `punarvaani demo` output if screen sharing |

**Everyone should know these five facts by heart:**
1. Patients after throat/oral cancer surgery, stroke or ALS can move their lips but cannot speak.
2. Punarvaani reads lips from a phone camera and speaks **in Indian languages, in the patient's own voice**.
3. No system does this for any Indian language today. English research systems exist, but none restores the patient's own voice from silent lips.
4. The full software pipeline is already built and tested (25+ commits, 82 automated tests). Real Hindi data and GPU training are next.
5. Plan: Hindi first, then Kannada, Marathi, Bengali, Tamil.

---

## 2. Call flow (what to say, in what order)

The order is designed to convince: **problem → why now → what we already built → why it's first in the world → plan → what we need from him**. Ask for his help only after he has seen that the work is serious.

### 0:00–1:00 · Opening (1 min)
> "Thank you for your time, sir. In 15 minutes I'd like to show you what Punarvaani is, what we have already built, how it fits the ICMR First in the World grant, and what we would need from you to apply."

*Why:* sets an agenda and a time limit, which shows respect for his schedule.

### 1:00–3:00 · The problem (2 min)
- People who lose their voice to **laryngectomy (throat cancer surgery)**, stroke or ALS can still move their lips, but no sound comes out.
- India has one of the highest burdens of oral and throat cancer in the world.
- Today's options are poor:
  - an **electrolarynx** sounds robotic;
  - writing or typing is slow;
  - **voice banking** (recording your voice before surgery) is rarely used: in one study only 18 of 30 patients recorded enough and only 7 actually used it.
- One line to land it: *"Surgery took their voice. Punarvaani gives it back."*

### 3:00–6:00 · What Punarvaani does, and what we already built (3 min)
- A phone camera watches the lips. The system speaks for the person **in Hindi**, **in their own voice**, rebuilt from old WhatsApp voice notes and family videos.
- Two ways of producing speech:
  - **Path A:** lips → Hindi text → corrected by an AI language model → spoken in the cloned voice. Reliable.
  - **Path B:** lips → speech sounds directly → voice. More natural rhythm and emotion. This is the research contribution.
- **Phrase mode** as a safety net: the patient's 200–500 everyday phrases ("मुझे पानी चाहिए", "दर्द हो रहा है") are recognised with high reliability. If unsure, it stays silent rather than saying the wrong thing.
- **Already done** (this is the strongest point; say it confidently):
  - the complete pipeline is coded and tested (82 automated tests);
  - a data engine to collect Hindi lip-reading data from public videos;
  - voice rebuilding from noisy voice notes, including removing other family members' voices;
  - personalisation experiment: on simulated data, adapting to one person's silent mouthing cut errors from 92% to about 15% with only 64 short clips.
- Be honest: *"These results are on simulated data. They prove the pipeline works. Real Hindi results come once we train on real data with GPUs."*

### 6:00–8:00 · Why this is "first in the world" (2 min)
ICMR's first selection step is: *is it really first in the world?* Five things no existing system combines:
1. **Indian-language** lip-to-speech (no large Hindi lip-reading dataset even exists, so we build one);
2. **personalised** to one patient from 10–15 minutes of video;
3. the patient's **own voice rebuilt from scraps** (old voice notes), not studio recordings;
4. works on **silent mouthing** (every existing dataset is people speaking aloud);
5. **phone camera only**, real time, no sensors on the face.

### 8:00–9:30 · Multilingual plan (1.5 min)
- This is his suggestion, so credit him: *"As you suggested, sir..."*
- Hindi first. Once it performs well, extend to **Kannada, Marathi, Bengali, Tamil** and others.
- The system is built to be language-independent. Each new language mainly needs data and fine-tuning:
  - the data engine works on any language's videos;
  - lip movements are largely shared across languages, so most of the model is reused;
  - voice cloning (IndicF5) already supports several Indian languages.
- This makes the impact national, which strengthens the "global impact" angle of the grant.

### 9:30–12:00 · ICMR plan (2.5 min)
- Track: **Proof of Concept**, up to **₹1 Crore**, **2 years**. The prototype (₹4 Cr) and final product (₹8 Cr) tracks can follow later.
- Deadline: **30 November 2026, 5:30 PM**, via the ICMR e-PMS portal. Results expected 31 December 2026.
- Year 1: Hindi data, the Hindi model, personalisation, a working demo. Year 2: Path B, two more languages, and a small validation with patients under ethics approval.
- Deliverables: a new Indian-language lip-reading dataset, a working phone demo, a pilot validation, and publications.

### 12:00–14:00 · The asks (2 min)
Ask clearly, one at a time:
1. *"Would you be willing to be the **PI** for this proposal, sir?"*
2. *"Can you suggest a faculty member from our institution to be the **Co-PI**? ICMR makes that mandatory."*
3. *"Do you know an ENT or oncology doctor who could be a clinical collaborator?"*
4. *"Can our institution provide the ethics committee approval and the institutional undertaking ICMR needs?"*
5. *(Optional)* GPU access through the department.

### 14:00–15:00 · Close with next steps (1 min)
> "If you agree, we'll send you a first draft of the proposal by [date: about 10 days from the call] for your review, and finalise it by 25 November so we submit before the deadline with a buffer. Does that work, sir?"

Then thank him and confirm who sends what, and by when.

**Tips for the call**
- Have the GitHub repo open; show `docs/ROADMAP.md` if he asks how organised the plan is.
- Keep every answer under 30 seconds unless he asks for detail.
- If you don't know something, say *"I'll check and get back to you today"*. Never guess.
- Take notes; send a short summary message after the call.

---

## 3. Slide deck outline

12 slides for the team discussion (and for sir later if he wants a deck). Each slide: title, bullet points, and speaker notes.

### Slide 1 · Title
- **Punarvaani (पुनर्वाणी): the voice, returned**
- Restoring a patient's own voice from silent lip movements, in Indian languages
- Team names · Institution · Guide: Sunag sir
- *Notes:* one sentence: "We read lips from a phone camera and speak in the patient's own voice."

### Slide 2 · The problem
- Throat/oral cancer surgery, stroke and ALS take away the voice; lip movement often remains
- India: one of the highest burdens of oral and throat cancer
- Current options: electrolarynx (robotic), writing (slow), voice banking (rarely done: 7 of 30 patients used it in one study)
- *Notes:* make it human: imagine not being able to tell your family you're in pain.

### Slide 3 · Our solution
- Phone camera → lip reading → speech in the patient's **own** voice
- Voice rebuilt from old WhatsApp voice notes and family videos
- Hindi first, then other Indian languages
- *Notes:* "Surgery took their voice. Punarvaani gives it back."

### Slide 4 · How it works (architecture)
- Camera → stabilised lip crop → visual speech model (AV-HuBERT, adapted to Hindi)
- **Path A:** lips → text → AI correction → cloned voice (reliable)
- **Path B:** lips → speech sounds → voice (natural)
- **Phrase mode:** everyday phrases, high reliability, stays silent when unsure
- *Notes:* use the diagram from the README; don't go deep on the technology unless asked. Section 5 has the libraries and models for a backup slide.

### Slide 5 · Why it's first in the world
- Indian-language lip-to-speech: no large dataset or system exists
- Personalised from 10–15 minutes of a patient's video
- Own voice from noisy old recordings
- Works on silent mouthing (all datasets are spoken aloud)
- Phone camera only, real time
- *Notes:* this slide answers ICMR's first selection question directly. Section 6 has the related-work table and the research question.

### Slide 6 · What's already built
- Complete pipeline coded and tested: 25+ commits, 82 automated tests, CI
- Hindi data engine, voice rebuilding, phrase mode, AI correction, real-time streaming
- Early result (simulated data): per-user adaptation cut errors from 92% to ~15% with 64 clips
- *Notes:* say clearly that results are on simulated data and real Hindi training is next.

### Slide 7 · Accuracy: the main risk, and how we handle it
- Lip reading is hard: many sounds look identical (प/ब/म)
- Four layers: personalisation → language model → AI context correction → phrase mode
- Phrase mode guarantees the most important sentences work even if open-vocabulary accuracy is limited
- *Notes:* naming the risk ourselves builds trust.

### Slide 8 · Multilingual expansion
- Hindi → Kannada, Marathi, Bengali, Tamil, …
- Same pipeline; new language = new data + fine-tuning
- National impact: patients across India, in their mother tongue
- *Notes:* credit sir's suggestion.

### Slide 9 · ICMR First in the World grant: the fit
- Track: **Proof of Concept**, up to ₹1 Crore, 2 years
- Deadline: **30 Nov 2026, 5:30 PM** (ICMR e-PMS portal); results 31 Dec 2026
- Needs: PI + **mandatory Co-PI from the same institution**; ethics (IEC) and institutional undertaking at funding time
- *Notes:* the later tracks (₹4 Cr prototype, ₹8 Cr product) give a path to scale.

### Slide 10 · Two-year plan
- **Year 1:** Hindi data (50–100 h) + volunteer recordings → Hindi model → personalisation → working demo
- **Year 2:** Path B + hybrid → 2 more languages → phone app → pilot validation with patients (ethics approved)
- Deliverables: Indian-language lip-reading dataset, phone demo, pilot study, publications
- *Notes:* point to `docs/ROADMAP.md` for the detailed checklist.

### Slide 11 · Team and roles
- PI: Sunag sir (to confirm) · Co-PI: [name] · Clinical collaborator: [ENT/oncology doctor]
- Data lead · Vision model lead · Speech model lead · System lead
- *Notes:* fill in names after the team meeting.

### Slide 12 · Asks and next steps
- PI / Co-PI confirmation · hospital collaborator · ethics and institutional support · GPU access
- Draft proposal to sir by [date] · final by 25 Nov · submit by 28 Nov (buffer before 30 Nov)
- *Notes:* end with a clear "yes/no" question.

---

## 4. Questions he may ask (and answers)

**"How is this different from existing lip-reading research?"**
Existing systems are English-only, use people speaking aloud, and produce a generic voice. We do Indian languages, silent mouthing, and the patient's own voice from old recordings, personalised per patient.

**"How accurate is it?"**
On real Hindi data we don't know yet; that's what the grant funds. English state of the art is around 25–30% word error. Our safety net is phrase mode, plus personalisation and AI correction. The simulated experiments show the methods work.

**"Where will the data come from?"**
Public Hindi news and lecture videos, mined automatically (50–100 hours) with quality checks, plus 5–10 volunteers recorded with consent, speaking and silently mouthing.

**"What about patient data and ethics?"**
We develop with healthy volunteers first. Patient involvement comes only in year 2, under ethics committee approval, with a hospital partner. No faces or voices are published without consent, and there's an offline mode so private sentences need not leave the device.

**"What do you need the money for?"** *(categories; amounts to be finalised with sir per ICMR norms)*
Research fellows (manpower), GPU compute, data collection and participant compensation, the hospital pilot study, travel/dissemination, contingency and institutional overhead.

**"Is the model ours? Licensing?"**
The code is ours. Some base models (for example Meta's AV-HuBERT) are licensed for non-commercial research, which is fine for a research grant. For a later product stage we'd retrain or replace those parts.

**"Can it run on a phone?"**
Phase 1: phone camera + laptop/edge GPU, target under 1 second. Later: compressed models directly on the phone.

**"Why you / is the team capable?"**
The full pipeline is already built and tested, so we are not starting from zero. The repo and roadmap show the plan in detail.

---

## 5. Libraries, models and tools

What we use at each stage of the architecture, what each one lets us do, and a one-liner you can say if asked. Use this for slide 4 (architecture) or as a backup slide.

### Pretrained models (we adapt these, we don't train them from scratch)

| Model | Stage | What it lets us do | Say it like this |
|---|---|---|---|
| **MediaPipe Face Landmarker** (Google) | [1] Lip crop | Finds 478 points on the face in every video frame, in real time, even on a phone | "Google's face-tracking model finds the lips in every frame." |
| **AV-HuBERT** (Meta) | [2] Lip reading | A model already trained on thousands of hours of video to connect lip movements with speech sounds; we adapt it to Hindi | "We start from Meta's lip-reading model instead of training from zero." |
| **IndicF5** (AI4Bharat) | Path A voice | Text-to-speech for Indian languages that copies a voice from a ~10-second sample (zero-shot voice cloning) | "Speaks in the patient's voice from just a few seconds of an old recording." |
| **mHuBERT-147** | Path B | A speech model trained on 147 languages, including Hindi; turns audio into "speech sound tokens" every 20 ms | "Turns speech into small sound units that the lip model learns to predict." |
| **IndicWhisper / Whisper** (AI4Bharat / OpenAI) | Data engine, voice rebuild | Speech-to-text with word timings; transcribes news videos and voice notes automatically | "Automatically writes down what is said, with timings, so we don't label by hand." |
| **ECAPA-TDNN** weights (SpeechBrain) | Voice rebuild | Makes a "voice fingerprint" so we can tell the patient's voice apart from relatives in family videos | "Recognises whose voice is whose." |
| **SyncNet** | Data engine | Checks whether lips and sound in a video match in time; filters dubbed or out-of-sync clips | "Throws away videos where the voice doesn't match the lips." |
| **Claude** (Anthropic) | Path A correction | A large language model that fixes lip-reading mistakes using context (प/ब/म look identical on lips) | "An AI language model fixes look-alike mistakes using context." |

### Python libraries

| Library | What it lets us do | Where we use it |
|---|---|---|
| **PyTorch** | The deep-learning framework: build neural networks, train them on GPUs | All models and training |
| **NumPy** | Fast maths on arrays of numbers | Lip-crop geometry, data processing |
| **OpenCV** | Read and write video, warp images, use the webcam | Lip crop, recorder, live demo |
| **MediaPipe** | Runs the face-landmark model | Lip crop |
| **torchaudio + soundfile** | Load, save and resample audio files | Audio front-end, voice rebuild |
| **Hugging Face Transformers** | Download and run pretrained models (IndicF5, Whisper, mHuBERT) with a few lines of code | TTS, speech recognition, speech units |
| **DeepFilterNet** | Removes background noise from speech | Cleaning old voice notes |
| **anthropic** (SDK) | Calls the Claude API | LLM correction |
| **PyYAML** | Reads the training config files | All training scripts |
| **jiwer** | Standard word-error-rate calculation; we use it to double-check our own metric code | Tests |
| **pytest** | Runs the 82 automated tests | Quality checks |
| **ruff** | Checks code style and catches common bugs | Quality checks, CI |

### Tools and platforms

| Tool | What it lets us do |
|---|---|
| **yt-dlp** | Download public Hindi news and lecture videos for the dataset |
| **ffmpeg** | Extract and convert audio from videos |
| **GitHub + GitHub Actions** | Version control; runs the tests automatically on every change (CI) |
| **Google Colab Pro / Kaggle** | Rented GPUs for training |
| **Hugging Face Hub (private)** | Stores our trained model checkpoints (never public data) |

### What we built ourselves (no library)

Worth mentioning: it shows depth, and it gives us full control.

| Built ourselves | Why not a library |
|---|---|
| **AV-HuBERT re-implementation** in plain PyTorch | Meta's original code (fairseq) is outdated and hard to install; ours loads their weights directly |
| **LoRA adapters** (small add-on layers for Hindi and per patient) | We needed two adapters at once (Hindi + patient) and swapping between patients |
| **Phrase mode** and **CTC beam search with a language model** | Our own decision logic, including staying silent when unsure |
| **HiFi-GAN vocoder**, **ECAPA network**, **YIN pitch tracker**, **k-means** | Adapted to our frame rates and to speaker conditioning |
| **Hindi text processing** (spelling cleanup, pronunciation rules, look-alike sound groups) | No existing library covers lip-reading look-alikes for Hindi |

**One-liner for the call:** *"We build on proven open models from Meta, Google and AI4Bharat, adapt them to Hindi and to each patient with small add-on layers, and wrote the parts specific to lip-to-speech ourselves."*

**If asked about licences:** AV-HuBERT is licensed for non-commercial research only, which is fine for a research grant. For a later product stage we would replace or retrain it. Check each other model's licence on its model card before the final proposal.

---

## 6. Related work, the real gap and our research question

Use this when someone asks *"Hasn't this already been done, at least in English?"* It is also the "first in the world" argument for ICMR.

### What already exists

| What exists | Examples | What it does | What it does **not** do |
|---|---|---|---|
| English lip-to-speech research | Lip2Wav (2020), lip2speech-unit (2023), LipVoicer (2024) | Turns lip videos into speech in research settings | Trained on people **speaking aloud**; outputs a generic or dataset voice, not the patient's own; English only; lab demos, not patient tools |
| English lip reading (lips → text) | Meta's AV-HuBERT on the LRS3 dataset | Reads English text from lip videos | No speech output; no Indian languages |
| Phrase-level patient apps | e.g. SRAVI by Liopa (UK), for ICU patients | Recognised a small set of English phrases | Phrase-only; not in the patient's own voice; English only |
| Voice banking | Recording one's voice before surgery | Patient-specific synthetic voice | Must be done **before** surgery; few patients do it (in one study only 7 of 30 used it) |
| Silent-speech interfaces | EMG sensors on the face/neck | Detect speech from muscle signals | Needs sensors on the face; impractical for daily use |
| Indian-language resources | vVISWa (Hindi/Marathi), IndicF5, IndicWhisper | Small word-level lip data; Indian TTS and ASR | No large Hindi lip-reading dataset; no Indian lip-to-speech system |

*Check the current status and exact details of each item in a proper literature search before the proposal; ICMR will verify the novelty claim.*

### The real gap

Each piece exists somewhere. **No system combines them, in any language:**

> Silent lip movements → phone camera only → speech in the patient's **own voice rebuilt from old recordings** → personalised to that patient → in **Indian languages** → with a phrase-mode safety net.

### Why Indian languages and not English

1. **Novelty is clear.** In English, reviewers can point to Lip2Wav and LipVoicer. For Hindi and other Indian languages there is no large lip-reading dataset or lip-to-speech system at all.
2. **The patients are here.** India has one of the highest burdens of oral and throat cancer; an English-only tool does not help most of them.
3. **The funder is Indian.** ICMR prioritises impact on Indian health.
4. **The dataset is a contribution in itself.** The first Hindi lip-reading dataset is a research output on its own.
5. **We build on the English work instead of repeating it.** AV-HuBERT, pretrained on English, is our starting point.

### Our novelty, in five points

1. First **Indian-language** lip-to-speech system, with the first large **Hindi lip-reading dataset** (built with our data engine).
2. **Silent mouthing**: we measure and close the gap between speaking aloud and silent mouthing; existing datasets only contain spoken speech.
3. **Voice from scraps**: the patient's own voice rebuilt from noisy WhatsApp notes and family videos, separating it from relatives' voices.
4. **Personalisation from minutes**: a per-patient adapter from 10–15 minutes of video.
5. **Safe by design**: phrase mode with rejection plus LLM context correction, so the system stays silent rather than saying the wrong thing.

### The research question

**Open-ended question:**
> *How can a person who has lost their voice speak again, naturally, in their own language and in their own voice, using only a smartphone camera, a few minutes of their silent lip movements, and whatever old voice recordings their family has?*

**Sub-questions (each can be a paper section):**
1. **Cross-lingual transfer:** how far can an English-pretrained visual speech model be adapted to Hindi with 50–100 hours of automatically mined video?
2. **Silent mouthing:** how much does accuracy drop when lips move without sound, and can a few minutes of a person's silent mouthing recover it?
3. **Voice from scraps:** how faithfully can a person's voice be rebuilt from 1–3 minutes of noisy, compressed archival audio, judged by speaker similarity and by their family?
4. **Context:** can a language model that knows the patient's situation correct errors between sounds that look identical on the lips?
5. **Practicality:** can the whole pipeline respond in under one second using only a phone camera?
6. **Scale:** how much new data does each additional Indian language need once Hindi works?

**One-liner for the call:**
> "English lip-to-speech exists as research, but only for people speaking aloud and in generic voices. No system restores a patient's own voice from silent lip movements in any Indian language. That is our gap, and we build on the English models rather than repeat them."

---

## ICMR call: key facts

| Item | Detail |
|---|---|
| Scheme | ICMR "First in the World" Challenge Grants (2026), investigator-initiated |
| Looks for | High-risk, high-reward, *first of its kind* biomedical technology |
| Proof of Concept | up to ₹1 Crore, max 2 years ← **our track** |
| Prototype development | up to ₹4 Crore, max 3 years |
| Final product / implementation | up to ₹8 Crore, max 4 years |
| Who can apply | Faculty and scientists at universities, medical colleges, recognised research organisations, NGOs, startups, MSMEs |
| Mandatory | One Co-PI from the PI's own institution |
| Submission | ICMR e-PMS portal only; incomplete proposals are rejected |
| Selection | (1) confirm "first in the world" potential → (2) shortlisting on relevance and impact → (3) committee discussion |
| Dates | Call released 1 Oct 2026 · portal opened 7 Oct 2026 · **last date 30 Nov 2026, 5:30 PM** · results 31 Dec 2026 |
| Funding | Released after institutional undertaking, IEC (as applicable) and bank details |

Read the full official notification before writing. The poster is a summary only.

---

## Proposal timeline to 30 November

| By | Milestone | Owner |
|---|---|---|
| Day of call | PI/Co-PI agreed; hospital contacts listed | Kartikey + sir |
| +3 days | Proposal outline and section owners fixed; read the full ICMR guidelines | All |
| +10 days | **First full draft to sir** | All (sections below) |
| 15 Nov | Sir's feedback incorporated; budget finalised; institutional letters requested | PI + team |
| 20 Nov | Final draft; institutional approvals in hand | PI |
| 25 Nov | Proposal frozen | All |
| **28 Nov** | **Submitted on e-PMS** (2-day buffer before the 30 Nov deadline) | PI |

**Proposal sections and suggested owners**
- Problem, clinical need and impact → System lead (with the clinical collaborator)
- Novelty ("first in the world") and state of the art → Vision model lead
- Methodology (data engine, models, voice rebuilding) → Data + Vision + Speech leads
- Preliminary work (what's built, simulated results) → Kartikey
- Work plan, milestones and deliverables → System lead (from `docs/ROADMAP.md`)
- Ethics, data protection → Data lead
- Budget and justification → PI with the team

---

## After the call

Send a short summary to sir the same day:

> Thank you for your time, sir. Summary: (1) [PI/Co-PI decision], (2) [hospital contact], (3) we'll share the first draft by [date]. Please let us know if we missed anything.

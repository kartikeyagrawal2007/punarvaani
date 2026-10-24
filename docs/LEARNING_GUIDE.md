# Punarvaani: the complete learning guide

*Everything that was built, how it works, why each decision was made, the background you need, and how to talk about it in an interview.*

---

## 0. How to use this guide

You did not write this code, so this guide is designed to let you **own** it. Read it in this order:

1. **§1 (one-page summary)** and **§2 (the 30-second and 2-minute pitches)**. Learn these by heart.
2. **§3 (architecture map)**, with the repo open next to it.
3. **§4 (component deep dives)**. For each component, read the section, then open the file and read its docstring. Every file starts with a long docstring that explains the *why*.
4. **§5 (decisions)** and **§6 (experiments and bugs)**. These are what interviewers dig into.
5. **§7 (background knowledge)**. Fill any gaps here before an interview.
6. **§8 (interview Q&A)**. Practise answering out loud.
7. **§9–§12**: how to run things, what is not done yet, glossary, papers.

Then run things yourself (§9). Change a number in a config, re-run, and see what happens. Nothing makes code yours faster than breaking it and fixing it.

> **Honesty note for interviews.** This code was written with an AI coding assistant. That is normal now, but be ready to say so if asked, and make sure you can explain every design decision in this guide in your own words. The decisions, experiments and bug stories below are what make the work defensible. Practise them until you can explain them without notes.

---

## 1. The project in one page

**Problem.** People who lose their voice to throat/oral cancer surgery (laryngectomy), stroke or ALS can often still move their lips, but no sound comes out. India has one of the highest burdens of oral and throat cancer in the world. Existing options are poor: voice banking (few patients do it before surgery), electrolarynx (robotic), or text-based communication (slow).

**Solution.** A phone camera watches the lips; the system speaks for the person **in Hindi**, **in their own voice**, rebuilt from old WhatsApp voice notes and family videos.

**Why it is new (the gap).** English lip-to-speech research exists (lip2speech-unit, LipVoicer). Nobody combines:

1. Hindi lip-to-speech (no big Hindi lip dataset exists, so we build a data engine to mine one);
2. personalisation to one person from 10–15 min of video (per-user LoRA);
3. a voice rebuilt from noisy scraps (denoise + speaker clustering + zero-shot cloning);
4. silent mouthing (datasets are all of people speaking aloud, and lips move differently when silent);
5. LLM context correction (पा/बा/मा look identical on the lips);
6. real time, phone camera only.

**Architecture in one picture.**

```
camera 25 fps ─► [1] lip crop (MediaPipe, stabilised 88×88) ─► [2] AV-HuBERT encoder (+hindi LoRA +user LoRA)
                                                                     │
           ┌─────────────────────────────────────────────────────────┴───────────────────────────┐
  PATH A "stenographer + newsreader"                                         PATH B "ventriloquist"
  CTC/attention → Hindi characters                                           50 Hz speech units (+mel, +pitch)
  → phrase mode (closed set, exact CTC likelihood, rejection)                → unit HiFi-GAN vocoder
  → else beam search + char LM → LLM/lexicon correction                        conditioned on the speaker embedding
  → IndicF5 voice cloning (reference clip)                                   (keeps rhythm, pauses, emphasis)
           └───────────── HYBRID: Path A text guides Path B units by CTC gradient ───────────────┘

Voice rebuild (offline, once): voice notes → denoise → segments → ECAPA embeddings → cluster speakers
                               → keep patient → speaker embedding (Path B) + reference clip (Path A)
```

**State of the work.** The whole pipeline is implemented as a tested Python package, `punarvaani/` (~6,400 lines, 81 tests). Every trainable part has been shown to learn on a synthetic lip-reading world on CPU. The real Hindi runs need: GPU time, the pretrained weights (AV-HuBERT, mHuBERT-147, IndicF5), mined data and volunteer recordings. Those are the next steps in the 16-week plan.

---

## 2. Your pitches

**30 seconds.**
> "Punarvaani gives a voice back to people who lost it to throat cancer surgery. A phone camera reads their lips and speaks for them in Hindi, in their own voice rebuilt from old WhatsApp voice notes. Technically it adapts Meta's AV-HuBERT visual speech encoder to Hindi with LoRA, then has two output paths: lips to text, corrected by an LLM and spoken by a voice-cloning TTS, and lips directly to speech units, turned into audio by a vocoder conditioned on the patient's voice. A closed-set 'phrase mode' guarantees the most important everyday phrases work even when open-vocabulary lip reading doesn't."

**2 minutes (add these).**
- *Data:* no Hindi lip dataset exists, so I built a mining pipeline: word-timestamped ASR → clip at pauses → filters (ASR confidence, Devanagari only, single shot, face size) → stabilised lip crop → SyncNet A/V-sync check → speaker-disjoint splits, with a per-reason filter report.
- *Accuracy risk:* lip reading is ~30% WER even in English, because many sounds look the same (visemes). I attack it at four levels: per-user adaptation, a character LM in beam search, LLM correction briefed on which letters are visually confusable, and phrase mode, which scores every known phrase by exact CTC likelihood and *rejects* when unsure, because saying nothing beats saying the wrong thing.
- *Personalisation result:* in a controlled experiment, a "silent-mouthing" user drives a model from 0% to 92% CER. A LoRA adapter trained on 64 clips recovers to 41% if placed only in attention/FFN layers, and to 15.5% if it also adapts the visual feature projections and BatchNorm. So the user-specific gap is low-level and visual, and that became the default recipe.
- *Voice:* family videos contain several voices, so voice rebuilding clusters ECAPA speaker embeddings and keeps the patient's cluster before cloning.

---

## 3. Architecture map: what lives where

| Stage | Folder / file | One line |
|---|---|---|
| Hindi text | `punarvaani/text/` | normalisation, 68-symbol character vocab, rule-based G2P with schwa deletion, visemes |
| [1] Lip crop | `preprocessing/lip_crop.py` | landmarks → smoothing → similarity alignment → 96×96 crop |
| Audio | `preprocessing/audio.py`, `pitch.py` | log-mel 100 Hz, VAD, YIN pitch, frame-rate contract |
| Data engine | `data_engine/` | mining, filtering, sync check, manifests, splits |
| Datasets | `datasets/` | manifest dataset, augmentation, dynamic batching, synthetic world |
| [2] Encoder | `models/encoder/` | AV-HuBERT re-implementation + fairseq loader + LoRA |
| Path A | `models/path_a/` | CTC/attention model, decoding, phrase mode, LLM correction, TTS, pipeline |
| Path B | `models/path_b/` | lip-to-unit model, k-means units, unit HiFi-GAN |
| Hybrid | `models/hybrid/` | CTC-guided unit refinement |
| Voice rebuild | `voice_rebuild/` | denoise, quality, ECAPA, speaker clustering, profile |
| Training | `training/` | generic trainer, three training entry points |
| Eval | `eval/` | metrics + evaluation runner |
| App | `app/` | streaming + live webcam demo |
| Recording | `recording/` | prompt selection, recorder, consent template |
| CLI | `cli.py` | `punarvaani demo / train-* / evaluate / rebuild-voice / mine` |

**The frame-rate contract** (memorise it; most AV bugs are frame-rate bugs):

| Signal | Rate | Per video frame |
|---|---|---|
| video | 25 fps (40 ms) | 1 |
| speech units (HuBERT stride 320 @ 16 kHz) | 50 Hz (20 ms) | 2 |
| mel / filterbank (hop 160) | 100 Hz (10 ms) | 4 |
| waveform | 16 kHz | 640 samples |

---

## 4. Component deep dives

### 4.1 Hindi text layer (`punarvaani/text/`)

**What.** Turns any Hindi transcript into one canonical spelling, maps it to model token IDs, and can convert it into phonemes and visemes.

**Normalisation** (`hindi.py`):
- **Unicode NFC.** Subtle point: the precomposed nukta letters (क़ U+0958 … य़ U+095F) are Unicode *composition exclusions*, so NFC actually *decomposes* them into base + nukta (U+093C). After NFC, "क़" is always two code points, however it was typed. Without this the same word has two encodings and two different targets.
- Remove zero-width joiner/non-joiner and BOM (they change rendering, not speech).
- **Numbers → words** in the Indian system (lakh, crore). Hindi 1–99 are irregular (इक्कीस is not "बीस एक"), so all 100 are listed. A lip reader cannot output "21"; it sees "इक्कीस" being said.
- Punctuation and Latin characters are dropped. Clips with lots of English (code-mixing) are filtered out by the data engine (`is_mostly_devanagari`).
- Two levels: `strict` (training targets, TTS) and `loose` (scoring: also merges nukta and chandrabindu→anusvara, which writers use inconsistently and lips cannot show). CER on `loose` measures reading errors, not spelling conventions.

**Vocabulary** (`vocab.py`): 68 symbols = 5 specials (`<blank>`=0 for CTC, `<pad>`, `<s>` as both BOS and EOS, `<unk>`, space) + 63 Devanagari letters, vowel signs and diacritics. See decision D3 for why characters and not subwords.

**G2P and schwa deletion** (`phonemes.py`): every Devanagari consonant carries an inherent "a" (schwa) that is often *not* pronounced: कमल = kamal (not kamala), कमला = kamlā. Rule (Ohala 1983): (1) delete the word-final schwa; (2) right to left, delete a medial schwa in the context V C _ C V. A consonant cluster (नमस्ते) blocks deletion. Anusvara becomes the nasal of the next consonant's place (अंक → a-ng-k, हिंदी → h-i-n-d-ii).

**Visemes.** A viseme is a set of phonemes that look the same on the lips. Voicing, aspiration and nasality happen in the throat and nose, which the camera cannot see, so प फ ब भ म are one viseme "P". Eight classes: P (bilabial), F (labiodental), T (tongue front, hidden), CH (palatal), K (back), A (open), I (spread), U (rounded). So पल / बल / मल are the same "movie" P-A-T. This module is used by phrase mode's text fallback, the lexicon corrector, the LLM prompt and the viseme error rate metric.

### 4.2 Lip crop, stage [1] (`preprocessing/lip_crop.py`)

**What.** Video frames + face landmarks → stable 96×96 grayscale mouth crops. This reproduces the LRS3/AV-HuBERT preprocessing, using MediaPipe Face Mesh (478 points) instead of dlib's 68.

**How.**
1. Interpolate landmarks on frames where the face was lost; reject the clip if more than 10% of frames are missing.
2. **Temporal smoothing**: a centred moving average over 12 frames (0.48 s). Crop jitter looks like lip motion to the model.
3. **Similarity alignment** (rotation + uniform scale + translation, Umeyama's closed-form least-squares solution via SVD) from rigid points (4 eye corners + nose tip) onto a canonical 256×256 face. This normalises head pose and camera distance.
4. Warp straight to a 96×96 window centred on the smoothed mouth (one warp instead of warp then slice).
5. Grayscale. Training then takes a random 88×88 crop (centre crop at test time), the size AV-HuBERT was pretrained on.

**Key insight to say in interviews:** *mouth corners are deliberately not used for alignment*. They move when you speak, so aligning on them would normalise away the very motion we want to read.

Also: `mouth_opening()` (inner-lip gap ÷ mouth width, scale-invariant), and `MediaPipeLandmarker`, which wraps the MediaPipe **Tasks** `FaceLandmarker` API in VIDEO mode. Newer MediaPipe versions removed the old `mp.solutions` API.

### 4.3 Audio front-end (`preprocessing/audio.py`, `pitch.py`)

- **Log-mel** implemented directly on `torch.stft` with our own mel filterbank (HTK mel scale, Slaney area normalisation), so results don't depend on the torchaudio version. Defaults (n_fft 640, hop 160, 80 mels, 16 kHz) follow lip2speech-unit / HiFi-GAN @16 kHz.
- `stack_frames`: 100 Hz → 25 Hz by stacking 4 frames (AV-HuBERT's audio input is 26 fbank × 4 = 104-d).
- **Energy VAD** with three conditions: an absolute floor, *the noise floor (10th-percentile frame energy) + 6 dB*, and within 30 dB of the loudest frame. The noise-floor condition was added after a test showed steady background noise being labelled as speech (bug story B4).
- **YIN pitch** (`pitch.py`): the difference function d(τ), cumulative-mean normalisation, first dip below a threshold, then parabolic interpolation. Vectorised with FFT cross-correlation. 20 ms frames so it lines up with speech units. `pitch_targets` normalises log-F0 per speaker, so the lip model predicts *relative* intonation and the vocoder maps it into the patient's own pitch range.

### 4.4 Data engine (`data_engine/`)

**Why.** There is no large Hindi lip-reading dataset, so we mine 50–100 h from Hindi news and lectures. This is itself a contribution.

**Per video:** download (yt-dlp) → 25 fps frames + 16 kHz audio (ffmpeg) → ASR with word timestamps → `segment_words` → text filters → single-shot check → landmarks → lip crop → sync check → save `.npy` crop, `.wav` and a manifest row.

- `segment.py`: clips of 1–12 s that start and end in pauses (a word cut in half at a clip edge is a guaranteed label error for CTC). 120 ms padding around each clip, because lips start moving before sound starts.
- `tracking.py`: shot cuts from colour-histogram total-variation distance; an IoU face tracker (faces barely move between 40 ms frames) that never crosses a cut; track filters for face size (tiny faces give mush crops) and *competing faces* (two large faces: who is talking?).
- `sync.py`: **SyncNet** idea. Embed 0.2 s of mouth video and 0.2 s of audio in a shared space. For offsets δ ∈ ±15 frames compute the mean distance; **offset** = argmin, **confidence** = median − min (a sharp minimum means the face we see is the voice we hear). Drop clips with low confidence (dubbing, voice-over B-roll); fix small offsets by shifting audio. Includes a small two-stream SyncNet and an InfoNCE training loss; production should use pretrained `syncnet_v2` weights.
- `pipeline.py`: `MiningPipeline` + **`FilterReport`** counting every dropped clip by reason. *Almost all the work in a data engine is filtering.* If 80% of a channel's clips fail sync, the channel is dubbed, so drop the channel rather than tuning thresholds.
- **Speaker-disjoint splits** by hashing the speaker/channel ID. Clips of the same anchor in train and test would measure memorisation of that anchor's lips. Hashing is deterministic and stable as data grows (new clips of old speakers stay in their split).

### 4.5 Visual speech encoder, stage [2] (`models/encoder/`)

**AV-HuBERT in one paragraph.** HuBERT (Hsu et al. 2021) learns speech representations without transcripts: cluster audio features with k-means, mask spans of the input, and train a transformer to *predict the cluster IDs of the masked frames*. AV-HuBERT (Shi et al., ICLR 2022) does this with audio *and* video input but audio-derived targets, so the visual stream is forced to learn which lip movements correspond to which speech sounds. That is exactly what lip reading needs, learned from unlabelled video. It is pretrained on English (LRS3 + VoxCeleb2). We adapt it instead of training from scratch, which would need thousands of hours and far more compute.

**Our implementation** (written from scratch, with **module names matching Meta's fairseq code** so the official checkpoints load by name):

```
video (B,1,T,88,88) ─► feature_extractor_video.resnet ─► (B,T,512) ─► .proj ─► (B,T,D)
audio (B,T,104)     ─► feature_extractor_audio.proj                ─► (B,T,D)   [zeros for lip reading]
concat (B,T,2D) ─► layer_norm ─► post_extract_proj (2D→D) ─► [time mask] ─► transformer ─► (B,T,D)
```

- **Front-end** (`resnet.py`): a 3D conv (5×7×7, stride 1×2×2) + max-pool, then a 2D ResNet-18 per frame, then global average pooling → 512-d per frame. The **3D stem's 5-frame temporal kernel** is what lets it see motion. One frame can't distinguish an opening mouth from a closing one. PReLU activations, as in AV-HuBERT.
- **Transformer** (`transformer.py`): wav2vec 2.0-style. **Convolutional positional embedding** (a wide grouped 1-D conv, kernel 128 ≈ 5 s, with weight norm) instead of sinusoids: it gives relative position and generalises to any length. **Separate q/k/v/out projections** (not PyTorch's fused `in_proj_weight`), so names match fairseq and LoRA can wrap exactly the projections we want. Pre-LN (Large) or post-LN (Base). LayerDrop.
- **Audio branch kept but fed zeros.** AV-HuBERT was trained with *modality dropout*, so it has seen video-only inputs, and video-only is how it is evaluated for lip reading. Keeping the branch keeps checkpoints loadable and allows audio-visual training later.
- **Presets**: `large` (1024-d, 24 layers, **324.6 M params**, of which the ResNet front-end is 11.2 M), `base`, `small`, `tiny` (64-d, 2 layers, for CPU tests).
- **Checkpoint loading** (`convert_fairseq_state_dict`): strips wrapper prefixes from fine-tuned checkpoints (`w2v_encoder.w2v_model.`), maps old-style weight norm (`weight_g`/`weight_v`) to PyTorch parametrisations (`original0`/`original1`), and drops pretraining-only tensors (`final_proj`, `label_embs_concat`). Tested by round-tripping a fake fairseq checkpoint.
- **Padding correctness**: padded frames are zeroed *before* the 3D stem, so a padded sample in a batch gives exactly the same features as running it alone (bug story B2).

### 4.6 LoRA (`models/encoder/lora.py`)

**Math.** Freeze the pretrained weight W (d_out × d_in) and learn a low-rank update:

  y = W x + (α / r) · B (A x),  A: r × d_in, B: d_out × r, r ≪ d

B starts at **zero**, so training starts *exactly* at the pretrained model (A gets Kaiming init). α/r scales the update so changing r doesn't require re-tuning the learning rate. After training, ΔW = (α/r)·B·A can be **merged** into W, so inference costs nothing extra.

**Numbers on our large model:** LoRA r=16 on q/v/out/fc1/fc2 + the two feature projections = **6.4 M parameters (1.5% of 425 M)**. With the new CTC head and 6-layer decoder trainable as well, 107 M (25%) train. The decoder is the big part, which is why `decoder_layers` is the first knob to reduce on small data.

**Why our own implementation (decision D5):** we need **named, stackable adapters**: `hindi` (language) learned once, then `user_<id>` (person + silent articulation) on top. One frozen base serves every patient. A patient's adapter is a few MB, swappable at runtime. Functions: `inject_lora`, `mark_trainable`, `set_active_adapters`, `merge_adapter`/`unmerge_adapter`, `save_adapter`/`load_adapter`, `merge_and_unload` (fold everything in and restore plain module names, so Path B can start from the adapted Path A encoder).

### 4.7 Path A recogniser (`models/path_a/model.py`, `decoding.py`)

**CTC (Connectionist Temporal Classification), the key idea.** The video has T frames; the text has L < T characters. We don't know which frame corresponds to which character. CTC adds a *blank* symbol and lets the model output one symbol per frame. A frame-level path like `--कक-म--ल` collapses (merge repeats, then remove blanks) to "कमल". The probability of a text is the **sum over all paths** that collapse to it, computed efficiently by dynamic programming (the forward algorithm). The loss is −log of that sum. Properties: monotonic alignment (speech moves forward in time), frame-level posteriors, and conditional independence between output tokens (CTC alone can't learn spelling).

**Attention decoder.** A transformer decoder (causal self-attention over previous characters + cross-attention over encoder features) is a conditional language model: it fixes spelling, but on its own it can loop or stop early on hard inputs.

**Hybrid CTC/attention** (Watanabe et al. 2017): loss = λ·CTC + (1−λ)·attention (λ=0.3; label smoothing 0.1 on the decoder). CTC regularises the encoder toward monotonic alignment, and we get *both* scorers at test time. Tied input/output embeddings in the decoder.

**Decoding.**
- `ctc_greedy`: argmax per frame, collapse. Fast, but it ignores the sum over paths.
- **`ctc_prefix_beam_search`**: keeps, per candidate prefix, `p_blank` (all paths ending in blank) and `p_nonblank` (all paths ending in the last character). The split is needed because a repeated character needs a blank in between: from "क" ending in blank, emitting क gives "कक"; from "क" ending in क, emitting क stays "क". Everything is in log space with log-sum-exp. Pruning: only the top-k tokens per frame, plus a log-prob threshold. **Verified**: with a wide beam, each hypothesis's score equals PyTorch's exact CTC likelihood (test).
- **Shallow fusion with a character n-gram LM**: add λ·log P_LM(char | prefix) when a character is appended, plus a length bonus (LMs prefer short outputs). The LM is our own **interpolated Witten-Bell** n-gram (`CharNGramLM`): it trusts contexts seen often with few distinct continuations. Tested to normalise to 1 and to break a visual tie.
- **Attention rescoring**: score the n-best list with the decoder (teacher forcing) and re-rank by λ·CTC + (1−λ)·attention + LM. Simpler than joint CTC/attention beam search and very effective (WeNet does the same).

### 4.8 Phrase mode (`models/path_a/phrase_mode.py`): the guaranteed fallback

**Why.** Open-vocabulary lip reading is ~30% WER even in English. But a patient's daily needs are a known set ("मुझे पानी चाहिए", "दर्द हो रहा है", "डॉक्टर को बुलाओ"). Picking one of N known phrases is far easier than writing any sentence.

**How.** For each phrase y, compute the exact **log P(y | video)** = −CTC loss, with all phrases scored in **one batched `ctc_loss` call** (the posteriors are expanded N times). Then:
- **Bayes with a usage prior**: posterior ∝ likelihood × prior. The prior is Dirichlet-smoothed counts that grow each time the user (or a carer) confirms a phrase. Phrases the person often says win ties. Tested: पल vs बल (identical on the lips) resolves to whichever the user confirms more.
- **Rejection** (two tests): (1) the top posterior must be ≥ 0.5; (2) a **likelihood-ratio test** against free decoding. If the best single unconstrained CTC path is much more likely (per frame) than the best phrase, the person said something outside the set. *Saying nothing ("please repeat") beats saying the wrong thing* for a patient.
- **Text fallback** for text-only recognisers: rank phrases by **viseme edit distance**, so "भुझे बानी चाहिए" matches "मुझे पानी चाहिए" perfectly.

**Why CTC scoring rather than a classifier head (decision D7):** adding or editing a phrase needs **no retraining** (a family member adds "बेटी को बुलाओ" and it works immediately). It reuses the same model as open mode, and gives real likelihoods that combine with priors and support rejection.

**Demo result** (`punarvaani demo`): in-set utterances → accepted with posterior 1.00 in ~15 ms; out-of-set utterances → correctly rejected → open decoding reads them correctly.

### 4.9 LLM context correction (`models/path_a/llm_correction.py`)

- **`ClaudeCorrector`**: the system prompt explains lip-reading ambiguity, lists the **visually confusable letter groups** (from `phonemes.py`), and gives the person's setting, profile, frequent words and phrase list. The user message has the recent conversation and the n-best list with scores. It is instructed to *stay close to the visual evidence*: swap letters only within a viseme group, fix vowel length, never add ideas or answer the sentence. It requires a **JSON-schema structured output** `{text, chosen_index, confidence}`.
  - Model `claude-opus-5-5` at **low effort** (latency-bound use), server-side refusal fallback enabled, prompt caching on the system prompt (it only changes per user, so it caches well).
  - **Graceful degradation**: any failure (no network, auth, rate limit, refusal, unparseable output) falls back to the offline corrector. A patient's device must never go silent because of hospital Wi-Fi.
- **`LexiconCorrector`** (offline, deterministic): for each word, candidate known words whose **viseme sequence** is within a normalised edit distance, then a **Viterbi** over candidates scored by −10·visual cost + λ·log P_bigram(word | previous). It turns "भुझे बानी चाहिए" into "मुझे पानी चाहिए".
- **Latency**: the LLM runs only for open-vocabulary utterances. Phrase mode never waits for it.

### 4.10 TTS in the user's voice (`models/path_a/tts.py`)

**IndicF5** (AI4Bharat) is F5-TTS trained on Indian languages. F5-TTS is a **flow-matching** model (a cousin of diffusion: it learns a velocity field that transports noise to mel spectrograms) that does **zero-shot voice cloning**. Given a reference clip and its transcript, it continues "in that voice". So no per-user training is needed. The reference comes from voice rebuilding. Our wrapper follows the model card's call (`model(text, ref_audio_path=..., ref_text=...)` → 24 kHz audio).

`CachedTTS` **pre-renders all phrases offline**, so a recognised phrase plays instantly (latency = recognition only). `ToneTTS` is a deterministic stand-in for tests.

`PathAPipeline` (`pipeline.py`) wires it all and records per-stage latency: encoder → phrase match → (beam → rescoring → correction) → TTS.

### 4.11 Voice rebuilding (`voice_rebuild/`)

**Problem:** WhatsApp notes and family videos are noisy, compressed (Opus ~16–24 kbit/s), sometimes narrowband, and contain *other people*.

- **Denoising** (`denoise.py`): DeepFilterNet if installed (best quality), else a **spectral Wiener filter**. The noise spectrum is estimated from the quietest 10% of frames, and the gain G = max(1 − β·N/|X|², floor) is smoothed over time. The **gain floor** avoids "musical noise" artefacts, which a voice cloner would faithfully copy.
- **Quality report**: VAD-free **percentile SNR** (noise = 10th-percentile frame power, speech = 90th − noise; archival notes always have pauses), clipping ratio, **effective bandwidth** (telephone-band audio, around 4 kHz, makes a muffled clone) and speech duration, combined into one score.
- **ECAPA-TDNN** (`speaker.py`), the "voice fingerprint" network: 1-D convolutions over log-mel frames. **SE-Res2Blocks** (Res2Net multi-scale channel groups + squeeze-excitation + dilations 2/3/4), **multi-layer feature aggregation**, and **attentive statistics pooling** (a per-frame attention weight gives a weighted mean *and* std, so it focuses on informative frames). Trained with **AAM-softmax**: the true-class logit is s·cos(θ + m), which forces an angular margin, so embeddings separate cleanly under cosine similarity. Cepstral mean normalisation removes microphone colouring. 6.2 M params at full size.
- **Rebuild** (`rebuild.py`): denoise → VAD regions (merging gaps < 0.3 s) → 1.5–4 s segments → quality filter → embed → **average-linkage agglomerative clustering** with a cosine-distance threshold (no need to know the number of speakers) → keep the patient's cluster (closest to an *enrolment clip* if the family has one known clip, else the largest cluster) → outputs:
  - **speaker embedding** = quality-weighted mean of kept embeddings (Path B vocoder);
  - **reference clip** ≈ 10 s of the best kept segments + its transcript (IndicF5, Path A);
  - **clean segments** list (vocoder fine-tuning).

  *A birthday video has five voices. Averaging them all would make a voice that is nobody's.* That is why clustering matters.

### 4.12 Path B (`models/path_b/`): lips → speech, no text in between

**Speech units.** Cluster IDs (K≈1000) of a self-supervised speech model's features, here mHuBERT-147 (HuBERT trained on 147 languages, including Hindi), from a middle layer, which is the most phonetic. Units capture *what* is said and *its timing*, mostly not *who* is speaking. They are a "phonetic alphabet learned from audio". Predicting them turns a hard regression (video → waveform) into **classification per 20 ms frame**, which is much easier and gives intelligible speech (Choi, Kim & Ro, Interspeech 2023, lip2speech-unit). We fit our **own k-means** (`units.py`, k-means++ initialisation) on Hindi features so the inventory covers Hindi sounds (retroflexes, aspirates).

**`LipToUnitModel`** (`model.py`): the shared visual encoder (25 Hz) + heads:
- **unit head**: linear → 2× upsample to 50 Hz → small residual conv stack → softmax over K;
- **mel head**: 4 frames per video frame (100 Hz), L1 loss. lip2speech-unit found that predicting mel *and* units ("multi-target") helps both;
- **pitch head** (optional): normalised log-F0 + voiced/unvoiced per unit frame, so intonation comes from the lips instead of a flat robotic pitch;
- **auxiliary character CTC**: content supervision from the transcript.

**Alignment is free**: units come from the *same clip's* audio, so video frame t ↔ units 2t, 2t+1. No duration model is needed (unlike text-to-speech).

**Unit HiFi-GAN** (`vocoder.py`): a GAN vocoder. The **generator** embeds units (+ projected speaker embedding + pitch), then transposed convs upsample 5·4·4·2·2 = **320× (one unit = 20 ms at 16 kHz)**, with multi-receptive-field residual blocks. The **discriminators** are **MPD** (multi-period: reshape the waveform to 2-D with period p ∈ {2,3,5,7,11} to see pitch harmonics) and **MSD** (multi-scale: raw, ×2 and ×4 downsampled audio). **Losses**: LSGAN adversarial + 2·feature matching + 45·mel L1. The mel term stabilises early training; the GAN terms add the crispness that L1 alone blurs away.

**The key insight for the patient's voice:** the vocoder is trained on audio only (units extracted from audio → that audio). So we can **fine-tune it on the patient's cleaned voice notes with no video at all**. That is the step that makes Path B sound like them (`configs/vocoder_finetune.yaml`).

### 4.13 Hybrid A+B (`models/hybrid/guidance.py`), the stretch goal

Path B keeps natural timing but makes word errors; Path A gets words right but sounds flat. LipVoicer (ICLR 2024) combines them for diffusion with *classifier guidance*. We do it in the discrete unit space:

1. A small **unit ASR** model maps *soft* units (B, T, K) to character CTC posteriors. It is trained on audio only (units + transcripts), so data is plentiful.
2. At inference: p = softmax(z/τ) from Path B's logits z; L = CTC(unit_asr(p), Path A's text); update **z ← z − η·∇_z L**, plus a trust-region penalty λ·‖z − z₀‖² that keeps Path B's frame structure (timing).
3. argmax(z) → vocoder.

CTC alignment is monotonic and recomputed every step, so the text can only pull units *locally*: the "बानी" frames become "पानी" frames, while the pause before them stays where the lips put it. **Result on the toy test:** CTC loss 7.0 → 0.1 nats with only 10% of frames changed, and the leading silence untouched (bug/iteration story B6).

### 4.14 Real time (`app/`)

Offline preprocessing can look into the future; live, it can't.
- **`OnlineLipCropper`**: *causal* exponential moving average of landmarks and the mouth centre (a little lag instead of look-ahead). It holds the last face through short detection drop-outs.
- **`LipActivityDetector`**: end-pointing from **mouth motion** (mean absolute frame difference of crops), because *a silent speaker produces no audio, so a normal VAD is useless*. **Hysteresis** (separate on/off thresholds relative to an adaptive noise floor) plus a **hang-over** of 12 frames, so brief lip closures (प, म) don't split an utterance.
- **`UtteranceStreamer`**: a pre-roll buffer (lips move before the detector fires); calls Path A on each finished utterance.
- **`ChunkedUnitStreamer`** (Path B): every `chunk` frames, run the encoder on [left context | chunk | right look-ahead] and emit only the chunk's units. Each frame is emitted exactly once (tested). Latency ≈ (chunk + look-ahead) frames + compute, e.g. (8 + 4) × 40 ms ≈ 0.5 s.
- `realtime.py`: webcam → MediaPipe → crop → end-pointing → Path A → speaker. The phone version streams frames to the same `push_frame` over a WebSocket.

### 4.15 Training engineering (`training/`)

`Trainer` (`trainer.py`) has the features that matter for "one Colab GPU, big pretrained model, little data":
- **Mixed precision** (bf16 where supported, else fp16 + GradScaler), **gradient accumulation**, **gradient clipping** (transformers + CTC occasionally spike).
- **Warm-up then cosine decay**. Warm-up protects pretrained weights from large early updates while the new heads are still random.
- **AdamW parameter groups**: no weight decay on biases, norms, embeddings and LoRA B.
- **Frozen BatchNorm kept in eval mode.** A frozen ResNet in `train()` mode still *updates its BN running statistics* from our small batches and silently drifts away from the pretrained statistics. After every `model.train()` we put BN layers with frozen parameters back in eval. (When BN *is* adapted per user, its statistics update too, and that is AdaBN, intentionally.)
- Checkpoint/resume (Colab sessions die), keep best by validation metric, history JSON.

Entry points: `train_vsr.py` (synthetic / Hindi adaptation / personalisation, with LoRA injection, frozen adapters, adapter-only saving and `merged.pt`), `train_units.py` (Path B, can start from Path A's merged encoder), `train_vocoder.py` (alternating discriminator/generator steps on random 28-unit segments, AdamW β=(0.8, 0.99), exponential LR decay).

**Dynamic batching** (`datasets/av_dataset.py`, `DynamicBatchSampler`): batches are capped by *total frames*, not clip count (fairseq's `max_tokens`). Sorting within large random chunks keeps similar lengths together while staying random overall, so there is little padding and roughly constant GPU memory.

### 4.16 Evaluation (`eval/`)

- **WER/CER** at corpus level (sum of errors ÷ sum of reference lengths, not an average of per-utterance rates, which over-weights short clips), with substitution/deletion/insertion breakdown; checked against `jiwer`.
- **Viseme error rate (VER)**: CER on viseme sequences. *High CER but low VER* means the model sees the right mouth shapes and the remaining errors are look-alikes that context (LM/LLM) should fix. That diagnostic tells you where to spend effort.
- **Paired bootstrap** (Koehn 2004): resample test utterances with replacement, recompute both systems' WER, and count how often A is *not* better (a p-value) plus a 95% CI of the difference. Essential for ablations on small Hindi test sets, where a 1–2% WER difference can be noise.
- **MOS** with a bootstrap CI; **EER** for speaker verification ("is this X?" also gets a human blind test); **latency** p50/p90/p99 per stage.
- `evaluate.py`: greedy vs beam(+LM), phrase-mode top-1 accuracy, rejection rate and accuracy-when-accepted. Saves hypotheses so `compare()` can run the bootstrap between two runs. It also refuses to load a checkpoint whose adapter weights don't match the config (silently dropping them would evaluate the wrong model).

### 4.17 The synthetic world and the testing strategy (`datasets/synthetic.py`, `tests/`)

Real data is large, private and slow. **Bugs in shapes, masks, CTC lengths or LoRA wiring should be caught in seconds.** The synthetic world renders each letter of a small Devanagari alphabet as a *mouth shape* (an ellipse with letter-specific height, width and tilt), with 2–4 frames per letter, smooth transitions, a closed mouth between words, noise and jitter. It also emits Path B targets (unit IDs at 50 Hz, a fake mel, tone audio). Knobs:
- `confusable` pairs share a shape (built-in viseme ambiguity, to demo LM/prior effects);
- `open_scale`, `width_scale`, `tilt_offset` shift articulation style, which simulates the silent-mouthing / new-speaker gap.

*A model that can learn this has working plumbing; one that can't has a bug.* The 81 tests cover maths against references (CTC beam scores vs PyTorch's exact likelihood, WER vs jiwer, Umeyama recovery of a known transform, YIN on a known tone, LoRA merge exactness, k-means cluster recovery, fairseq key round-trip), invariances (padding, timing preservation), and end-to-end smoke runs of every training script. One `slow` test trains the tiny model and asserts CER < 35%.

### 4.18 Lip Lab: the live dashboard (`app/lab/`)

`punarvaani lab` opens a local web page: webcam → MediaPipe → mouth crop + lip-shape features → lip-activity end-pointing → two few-shot phrase readers, with a live score. It exists because the open-vocabulary model can't be judged on real lips until GPU training is done, but phrase mode on one person's face can be judged **today**.

- **Features, not pixels.** 40 lip landmarks mapped into the canonical face by the same similarity transform as the crop: 80 numbers per frame instead of 7,744 pixels, and invariant to head position, scale and tilt (tested). With 5 takes per phrase, a pixel model would overfit instantly.
- **Two readers on purpose.** DTW template matching (no training, works from one take) is the *baseline*; a temporal CNN with augmentation is the learned model. The dashboard uses whichever has the higher held-out accuracy. Always report a learned model next to a simple baseline.
- **Batched DTW.** The DP cell (i, j) depends on (i-1, j-1), (i-1, j) and (i, j-1), so cells on the same anti-diagonal are independent. Sweeping n + m diagonals over all templates at once gives the exact textbook result (tested) in ~50 ms for 60 takes.
- **Honest numbers.** Training reports leave-one-out (DTW) / 4-fold (CNN) accuracy: every take is predicted by a model that never saw it. The live test separates *accuracy*, *coverage* (how often it answered) and *accuracy when it answered*, because staying silent is a feature for patients, not a bug.
- **End-pointing without audio.** Silent speakers make no sound, so it watches lip-shape speed with hysteresis (start and stop thresholds) and a 0.4 s hang-over. A 1-second warm-up measures the noise floor first. Without it, the detector fired immediately and recorded 10-second junk takes, which dropped held-out accuracy from 100% to 44% in a test (bug B9).
- Every take is saved in the project manifest format (96×96 crops + lip points), so the lab doubles as the volunteer recording tool for personalisation.

---

## 5. Decisions log

### Big decisions

**D1. Adapt a pretrained English model (AV-HuBERT) rather than train from scratch.**
*Alternatives:* train a lip reader from scratch on Hindi; use an English lip-reading model as is.
*Why:* self-supervised AV pretraining learns lip ↔ sound correspondences that are largely language-independent (lip shapes for /p/, /a/, /u/ are universal); only the sequence/lexical level is language-specific. From scratch would need thousands of hours of video. *Trade-off:* the AV-HuBERT license is non-commercial research only.

**D2. Re-implement the AV-HuBERT architecture in plain PyTorch with fairseq-compatible names, instead of depending on fairseq.**
*Why:* fairseq is unmaintained and painful to install on modern Python/CUDA (Colab). Our version is ~400 lines you can read, LoRA-friendly (separate q/k/v), and still loads Meta's weights by name. *Risk:* subtle architecture mismatches. Mitigated by matching module names/shapes exactly (the large preset is 324.6 M params, as expected) and a conversion test. **First thing to verify on GPU:** load `large_vox_iter5.pt` and check the missing/unexpected key report is empty.

**D3. Character-level Devanagari targets, not subwords.**
AV-HuBERT's English VSR uses 1000 SentencePiece units, but it was fine-tuned on 433 h. With 50–100 h of Hindi and rich morphology, most subwords would be rare. ~70 characters are each seen thousands of times, need no tokenizer training, and keep the model open-vocabulary (names, new words). The inventory is **fixed** (not scanned from data), so checkpoints stay compatible across dataset versions.

**D4. Two output paths (A first, then B), plus phrase mode as a guaranteed fallback.**
Path A is reliable and gives a working demo by week 8; Path B is the research contribution (natural prosody). Phrase mode de-risks the whole project: even if open-vocabulary WER stays high, the most important sentences work.

**D5. Our own LoRA with named, stackable adapters (not `peft`).**
We need `hindi` + `user_<id>` adapters simultaneously, adapter-only save/load, runtime swapping between patients, merge for zero-overhead inference, and `merge_and_unload` for Path B. Writing it ourselves (~200 lines) makes this explicit and testable.

**D6. Per-user adapter placement: include the visual feature projections and BatchNorm, not just attention/FFN.**
Decided by experiment (§6, E2): attention/FFN-only LoRA recovers a silent user from 92% to 41% CER; adding the video/post-extract projections gives 18.9%; adding ResNet BatchNorm (affine + statistics, AdaBN-style) gives 15.5%. The person-specific gap is low-level and visual (mouth shape and size), so the adapter must reach the front-end.

**D7. Phrase mode scores phrases by exact CTC likelihood instead of training a classifier.**
No retraining when phrases change; same model as open mode; real likelihoods allow Bayesian priors and principled rejection.

**D8. Rejection over guessing.**
For a patient, a wrong sentence ("I'm fine" instead of "I'm in pain") is worse than "please repeat". Two tests: posterior confidence and a likelihood ratio against free decoding.

**D9. LLM correction is optional and always has an offline fallback.**
Hospital Wi-Fi fails; APIs time out. The LLM gets a viseme-confusion briefing and must output structured JSON. The lexicon/viseme corrector keeps the device talking offline.

**D10. Voice rebuilding clusters speakers before averaging.**
Family recordings contain several voices. A cosine-threshold agglomerative clustering + enrolment-clip matching isolates the patient.

**D11. Path B predicts units (classification), not waveforms or mels directly; vocoder personalised from audio only.**
Classification per 20 ms is far easier to learn from lips; units are speaker-independent, so identity comes from the vocoder's speaker embedding; and the vocoder can be fine-tuned on voice notes with no video.

**D12. Speaker-disjoint data splits.**
Otherwise test accuracy measures memorising specific anchors' lips.

**D13. A synthetic world + a CPU test suite before any GPU run.**
GPU hours are scarce (Colab/Kaggle quotas). Every bug caught on CPU in seconds saves a wasted overnight run.

### Smaller decisions (know the reason for each)

| Decision | Reason |
|---|---|
| Align lips on eyes + nose, never mouth corners | mouth corners move with speech |
| Centred smoothing offline, causal EMA live | can't look into the future in real time |
| 88×88 random crop from 96×96 + horizontal flip | AV-HuBERT input size; speech is left-right symmetric; simulates tracking error |
| Zero padded frames before the 3D stem | batching must not change real frames' features |
| CTC weight 0.3, label smoothing 0.1 | standard hybrid CTC/attention settings |
| Attention *rescoring* instead of joint CTC/attention beam search | far simpler, nearly as good |
| Witten-Bell char LM | tiny, dependency-free, good with sparse counts |
| Time masking (p=0.3, span 5) + LayerDrop 0.1 when adapting | regularise on a small dataset |
| Warm-up 2000 steps | protect pretrained weights while new heads are random |
| Frozen BN in eval mode | prevent silent statistics drift |
| Weight decay off for norms/biases/LoRA-B | standard practice; decaying B pulls the adapter to zero |
| Dynamic batching by total frames | constant memory, little padding |
| Units from our own k-means on Hindi features | cover Hindi sounds; the English codebook may merge retroflex/dental |
| Speaker-normalised log-F0 targets | predict relative intonation; the vocoder applies the patient's range |
| HiFi-GAN on random 28-unit segments | uniform batches, fast discriminators (as in speech-resynthesis) |
| Percentile SNR (VAD-free) | VADs are fooled by steady noise (bug B4) |
| Gain floor in the Wiener filter | avoid musical noise that a cloner would copy |
| Pre-render phrase audio | phrase mode latency = recognition time only |
| Claude at low effort, structured output, prompt caching | latency-bound; parseable; system prompt is per-user stable |
| Hysteresis + hang-over in lip-activity detection | don't split words at brief closures |
| Greedy diphone set cover for volunteer prompts | maximum coverage of mouth transitions per minute recorded |
| Record spoken then silent for each prompt | measure the silent gap; spoken audio can supervise silent takes |
| `zlib.crc32` seed for the recorder shuffle | Python's `hash()` changes between runs (bug B5) |

---

## 6. Experiments, results and bug stories

All numbers below are from the **synthetic** world on CPU. They show that the methods work and how they compare. They are **not** Hindi results. Say this clearly in interviews.

### E1. Path A learns (`configs/synthetic_vsr.yaml`)
Tiny model (333 K params): CER 5.8% after 100 steps, 0.26% after 200, **0.0% after 300–400** (≈1.5 min on 4 CPU cores).

### E2. Personalisation for a "silent-mouthing" user (`python scripts/ablation_personalization.py`)
The user's mouth openings are 55% as large, 15% narrower and tilted (`open_scale=0.55, width_scale=0.85, tilt_offset=0.35`). The base model (0% CER on normal speakers) gets **92.1%** on this user. A LoRA adapter trained on **64 clips for 200 steps**:

| LoRA placement | Trainable params | CER after |
|---|---|---|
| attention + FFN only | 6.1 K (1.8%) | 41.3% |
| + video / post-extract projections | 8.1 K (2.4%) | 18.9% |
| + ResNet BatchNorm (AdaBN/BitFit-style) — **default** | 9.1 K (2.7%) | **15.5%** |

*Interpretation:* the domain gap is in low-level visual features, so the adapter must reach them. This is a hypothesis to confirm on real silent-mouthing data in weeks 9–10.

### E3. Path B learns units (`configs/synthetic_units.yaml`)
100% frame unit accuracy, 0% unit error rate after 100 steps (the toy is easy because units are frame-aligned letter IDs).

### E4. Vocoder trains (`configs/synthetic_vocoder.yaml`)
For a tiny generator, validation mel L1 falls from 0.98 (step 100) to 0.68 (step 200); training mel L1 was 1.18 at step 50.

### E5. Phrase mode with rejection (`punarvaani demo`)
6 in-set utterances → all accepted at posterior 1.00 (~15 ms each). 6 out-of-set → all rejected by the likelihood-ratio test, then decoded correctly by open-vocabulary beam search.

### E6. Hybrid guidance (`tests/test_hybrid.py`, `configs/synthetic_unit_asr.yaml`)
The unit ASR itself reaches 0% CER on the toy world in 100 steps (`punarvaani.training.train_unit_asr`).
Path B's guess spells a wrong middle "word"; after 50 guidance steps the CTC loss of Path A's text falls from 7.0 to 0.1 nats, only 10% of frames change, and the leading silence (timing) is untouched.

### Bug stories (great interview material: what broke, how it was found, the fix)

**B1. `3e-4` read as a string.** PyYAML follows YAML 1.1, where a float needs a dot (`3.0e-4`). `lr: 3e-4` silently became the *string* "3e-4". Found by a config test doing arithmetic. Fix: a SafeLoader subclass with a float resolver that accepts `3e-4`.

**B2. Padding leaked into real frames.** A padded sample in a batch gave different features from running it alone, because the 3D stem's 5-frame kernel sees padded frames next to the last real frames. Found by a padding-invariance test. Fix: zero padded frames before the stem (matching the conv's own zero padding), so batching is exactly equivalent.

**B3. LoRA dropout on at inference.** Modules created by `inject_lora` default to training mode even inside an `eval()` model, so LoRA dropout was active at test time and outputs were random. Found by the merge-exactness test (merged ≠ unmerged). Fix: a new `LoRALinear` inherits the wrapped layer's train/eval mode.

**B4. VAD fooled by steady noise.** The energy VAD labelled background noise as speech, so SNR estimates were meaningless (a noisy file "scored" 35 dB). Found by a test asserting that denoising raises SNR. Fix: a noise-floor-adaptive VAD and a VAD-free percentile SNR.

**B5. Recorder resume shuffled prompts differently.** Python's string `hash()` is salted per process, so a resumed recording session mapped saved take indices to the wrong sentences. That is silent label corruption. Fix: `zlib.crc32` seed.

**B6. Hybrid guidance oscillated.** A globally normalised gradient step fought the trust-region term and spread edits everywhere (CTC stuck at 6.4). Plain gradient steps made small local edits (CTC 0.1, 10% of frames changed). Kept the experiment in the docstring.

**B7. Path B couldn't load Path A's encoder.** LoRA-wrapped checkpoints store `q_proj.base.weight` + adapter tensors, which don't match a plain encoder. Fix: `merge_and_unload` and a saved `merged.pt` (now loads with 0 missing keys, tested).

**B8. A failing test got committed.** A shell pipeline (`pytest | tail`) hid pytest's exit code, so the commit went through. Fix: a check script that fails on test failure; the commit was amended before pushing. (Lesson: check `PIPESTATUS`, or use `set -o pipefail`.)

**B9. The lab recorded 10-second junk takes on start-up.** Lip-activity end-pointing compares motion to a noise floor learned from recent frames. At start-up there is no history, so the floor sat at its minimum, normal jitter crossed the start line, and every take ran to the 250-frame cap. Held-out accuracy on the synthetic face fell to 44%. Fix: a `warmup` period (1 s) that only measures the floor, plus a *CALIBRATING* badge so the user keeps still. Back to 100%.

---

## 7. Background knowledge you must know

Each item: what it is → why it matters here → one line you can say.

**Self-supervised learning (SSL).** Learn representations from unlabelled data by predicting part of the input from the rest. *Here:* AV-HuBERT and mHuBERT are SSL models; units are SSL clusters. *Say:* "SSL lets us use thousands of hours of unlabelled video and audio; labels are only needed for the small fine-tuning step."

**HuBERT / masked prediction.** k-means cluster IDs as pseudo-labels; mask spans; predict the masked frames' IDs (BERT for speech); iterate the clustering on the model's own features.

**Transformer & attention.** softmax(QKᵀ/√d)V; multi-head; pre-LN vs post-LN (pre-LN is more stable for deep models); positional information (sinusoidal, learned, relative, convolutional).

**CTC** (§4.7). Blank, collapse, sum over alignments, forward algorithm, monotonic, conditionally independent outputs. *Pitfall:* the input length must be ≥ the target length (+ repeats); `zero_infinity=True` guards impossible pairs.

**Beam search / prefix beam search / shallow fusion / rescoring** (§4.7).

**n-gram LMs and smoothing.** Why smoothing (unseen n-grams); interpolation; Witten-Bell; Kneser-Ney (the usual best; mention you know it).

**LoRA** (§4.6). Low intrinsic rank of fine-tuning updates; zero-init B; α/r scaling; merging; where to place adapters (attention vs FFN vs front-end); related: adapters, prefix tuning, BitFit (bias-only), AdaBN (re-estimate BN statistics on the target domain).

**BatchNorm subtleties.** Running statistics update in train mode even when the weights are frozen; small batches give noisy statistics; eval mode uses running statistics.

**Mixed precision.** fp16 needs loss scaling (gradients underflow); bf16 has the fp32 exponent range, so no scaler is needed; keep softmax/CTC in fp32 (we call `.float()` before `log_softmax`).

**Visemes, co-articulation, silent speech.** Many phonemes per viseme. Co-articulation means neighbouring sounds change lip shapes. Silent mouthing is hypo-articulated (smaller movements) and loses airflow-related cues.

**Mel spectrograms.** STFT → power → triangular mel filters (perceptual frequency scale) → log. Hop/window sizes set the frame rate.

**Vocoders and GANs.** Generator vs discriminator; LSGAN loss; feature matching; why the mel loss stabilises; MPD/MSD; HiFi-GAN vs WaveNet (autoregressive, slow) vs diffusion vocoders.

**Flow matching / diffusion (F5-TTS, LipVoicer).** Learn to transform noise into data step by step; *classifier guidance* = add a classifier's gradient to steer generation (which is exactly our hybrid idea, in unit space).

**Speaker embeddings.** x-vectors → ECAPA-TDNN; AAM-softmax (ArcFace-style margin); cosine scoring; EER.

**k-means.** Lloyd's algorithm; k-means++ initialisation (pick each new centroid with probability ∝ squared distance) to avoid bad local minima.

**Pitch (F0).** Autocorrelation / YIN; voiced vs unvoiced; log-F0 normalisation per speaker.

**Similarity transforms & Umeyama.** Least-squares rotation + scale + translation via SVD of the cross-covariance; the reflection fix (det sign).

**Evaluation statistics.** Corpus-level WER; why per-utterance averages mislead; bootstrap confidence intervals; paired tests; MOS and its variance.

**Speech enhancement.** Spectral subtraction / Wiener filter; noise estimation; musical noise; neural enhancers (DeepFilterNet).

**SyncNet.** Two-stream contrastive embeddings; offset search; confidence = median − min.

**LLM prompting for correction.** Constrain the model to the evidence; structured output; fallbacks; latency/cost; privacy (patients' sentences go to an API: obtain consent, or use the offline corrector).

---

## 8. Interview question bank (with answers)

**Q: Why is lip reading so hard?**
Visemes: many phonemes look identical (p/b/m; t/d/n; k/g), voicing/aspiration/nasality are invisible, the tongue is mostly hidden, co-articulation blurs boundaries, and speakers differ. Even the best English open-vocabulary models are at ~25–30% WER.

**Q: Why AV-HuBERT and not a model trained on Hindi?**
There is no large Hindi lip-reading dataset. AV-HuBERT's self-supervised pretraining learns lip ↔ sound correspondences that transfer across languages. We adapt the sequence-level knowledge with a small Hindi dataset and LoRA.

**Q: Explain CTC to me like I'm an engineer.**
(§4.7) Per-frame outputs plus a blank; a text's probability is the sum over all frame paths that collapse to it, computed by the forward algorithm; it learns alignment for free, but assumes outputs are conditionally independent, which is why we add an attention decoder and an LM.

**Q: Why both CTC and attention?**
CTC gives monotonic alignment, stability and frame posteriors (needed for phrase mode and guidance); attention models character dependencies. Joint training regularises the encoder; at test time CTC proposes, attention re-ranks.

**Q: How does prefix beam search differ from normal beam search?**
It merges all alignments of the same prefix and tracks blank-ending and non-blank-ending probabilities separately, so repeated characters are handled correctly and the score is a sum over paths, not one path.

**Q: How does phrase mode work, and why not a classifier?**
(§4.8) Exact CTC likelihood for each phrase in one batched call, plus a usage prior and rejection. No retraining when phrases change, and it gives calibrated-ish likelihoods.

**Q: How do you decide to reject?**
The posterior must be ≥ a threshold, *and* the best phrase must explain the video nearly as well as free decoding (a per-frame log-likelihood ratio). If free decoding is much more likely, the person said something else.

**Q: What is LoRA and why use it here?**
(§4.6) Freeze W and learn B·A of rank r; ~1.5% of parameters; no overfitting on small data, small per-user files, and one base model for all patients.

**Q: Where would you put LoRA adapters, and how did you decide?**
By experiment (E2): for a person/articulation shift, the adapter needs to reach the low-level visual features (projections + BN), not just attention.

**Q: What is AdaBN / why adapt BatchNorm?**
BN statistics encode the input distribution; a new speaker or silent mouthing shifts it. Re-estimating BN statistics (and affine parameters) on the user's data is a cheap, strong domain adaptation.

**Q: How do you handle the "silent vs spoken" difference?**
Record each prompt spoken *and* silent; measure the gap; train the per-user adapter on silent takes (optionally both); time masking for robustness. Spoken takes provide audio that can supervise silent ones after alignment (future work: DTW on lip landmarks).

**Q: How do you get the patient's voice if they can't speak anymore?**
From archival audio: denoise, segment, quality-score, cluster speakers to remove relatives, then (Path A) zero-shot cloning with IndicF5 from a ~10 s reference + transcript, and (Path B) a speaker embedding plus vocoder fine-tuning on the cleaned notes. The vocoder needs only audio.

**Q: What are speech units and why use them for Path B?**
k-means clusters of self-supervised speech features at 50 Hz: phonetic and timing content, mostly speaker-independent. Predicting them is classification, which is far easier than regressing waveforms from lips, and the vocoder adds identity back.

**Q: How does HiFi-GAN work?**
(§4.12) A transposed-conv generator with MRF residual blocks; MPD + MSD discriminators; LSGAN + feature matching + mel L1.

**Q: What does the hybrid do?**
(§4.13) Gradient guidance of unit logits toward Path A's text through a unit-ASR CTC loss, with a trust region to keep timing. It is LipVoicer's classifier guidance in discrete unit space.

**Q: What metrics do you use, and how do you know an improvement is real?**
WER/CER/VER, phrase top-1 and rejection, speaker similarity/EER plus blind family tests, MOS, latency. Use a paired bootstrap for significance on small test sets.

**Q: Why viseme error rate?**
It separates "saw the wrong mouth shape" from "picked the wrong look-alike word", which tells you whether to work on the visual model or on context/LM.

**Q: How do you mine data, and what goes wrong?**
(§4.4) The main failure modes: dubbing/voice-over (sync check), cuts (shot detection), multiple faces, tiny faces, ASR errors (confidence filter), code-mixing (Devanagari filter), and leakage (speaker-disjoint splits). The filter report shows where clips die.

**Q: How do you make it real time?**
A causal crop; lip-motion end-pointing with hysteresis; Path A per utterance with pre-rendered phrase audio; Path B chunked with bounded look-ahead (~0.5 s). The LLM is skipped for phrases. Target < 1 s end to end.

**Q: What are the risks?**
Accuracy (mitigated by personalisation, LM/LLM, phrase mode), data volume/quality (data engine + volunteers), voice quality from scraps (clustering + denoising + quality scoring), ethics/privacy (consent, no public upload, offline mode), and the AV-HuBERT license (research only).

**Q: What would you do with more time / compute?**
Real Hindi training and the silent/spoken study; DTW-aligned cross-modal supervision for silent takes; joint CTC/attention beam search; a better LM (KenLM/Kneser-Ney, or a small neural LM); a learned unit ASR + full hybrid; on-device quantisation (int8) for the phone; a hospital validation study.

**Q: How do you know how well it works on a real person today?**
The Lip Lab (`punarvaani lab`): record ~5 takes per phrase, train a DTW baseline and a temporal CNN in about a minute on a laptop, and read held-out accuracy plus a live test score, silent vs spoken. It measures phrase mode on one face, not open-vocabulary Hindi lip reading, which needs the GPU-trained model.

**Q: What was the hardest bug?**
Pick B2 (padding leak) or B3 (LoRA eval mode) and tell it as: symptom → test that caught it → root cause → fix → what you'd do to prevent it.

**Q: How did you test ML code?**
Against references (exact CTC, jiwer, Umeyama, YIN), invariance tests (padding, timing), tiny end-to-end runs of every training script, and a learning test on a synthetic task.

**Q: Ethics?**
Consent template, no public data upload, mouth-only crops, deletion on withdrawal, an approved protocol for patients, and offline correction so private sentences need not leave the device.

---

## 9. How to run everything

```bash
# setup (Python ≥ 3.10)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ".[vision,audio,llm]" for real video/audio and Claude

# tests
pytest -q -m "not slow"            # ~10 s
pytest -q -m slow                  # tiny model learning test (~40 s)
ruff check punarvaani tests && ruff format --check punarvaani tests

# synthetic experiments (CPU)
punarvaani demo                                        # E1 + E5
python scripts/ablation_personalization.py             # E2
punarvaani train-units --config configs/synthetic_units.yaml       # E3
punarvaani train-vocoder --config configs/synthetic_vocoder.yaml   # E4
punarvaani evaluate --run outputs/synthetic_vsr --phrases configs/phrases_hi.txt

# live dashboard on your own lips (needs ".[vision]"), or a cartoon face without a camera
punarvaani lab --user yourname
punarvaani lab --synthetic

# override any config key from the command line
punarvaani train-vsr --config configs/synthetic_vsr.yaml train.max_steps=100 model.ctc_weight=0.3
```

Real runs (GPU) are in the README's "Real pipeline" section and `configs/README.md`.

**LLM correction:** `pip install anthropic`, set `ANTHROPIC_API_KEY` (or `ant auth login`), then pass `ClaudeCorrector(fallback=LexiconCorrector(corpus))` to `PathAPipeline(corrector=...)`.

---

## 10. What is NOT done yet (next steps, mapped to the 16-week plan)

| Week | Task | Code ready? | What's needed |
|---|---|---|---|
| 1–2 | Download AV-HuBERT Large; check `load_fairseq_checkpoint` reports no missing keys; run English inference | ✅ | GPU, weights |
| 3–5 | Choose Hindi sources (news/lectures, check terms of use), run `punarvaani mine`; train/obtain SyncNet + ECAPA weights; record 5–10 volunteers | ✅ | sources list, ASR checkpoint (IndicWhisper), volunteers + consent |
| 6–7 | `configs/hindi_adapt.yaml` on 50–100 h; char LM on Hindi text; voice rebuild for a test speaker | ✅ | GPU (~1–2 days on A100) |
| 8 | **Checkpoint:** Path A demo with phrase mode + LLM | ✅ | — |
| 9–10 | `configs/personalize.yaml` per volunteer; silent vs spoken gap study | ✅ | recordings |
| 11–12 | Extract mHuBERT units (+ fit k-means), `path_b_units.yaml`, vocoder pretrain + fine-tune | ✅ | GPU (vocoder pretraining is the longest job) |
| 13 | Train the unit ASR (`train_unit_asr.py`); evaluate the hybrid | ✅ | units + transcripts (audio only) |
| 14 | Phone → laptop streaming (WebSocket) | ⚠️ laptop side done | phone client |
| 15–16 | Evaluation, user study, paper | ✅ metrics | listeners, families |

**Known limitations to be honest about:**
- All results so far are synthetic. Real Hindi WER is unknown until weeks 6–8.
- The G2P rules are approximate for loanwords and compounds (fine for visemes, not perfect phonetics).
- The SyncNet/ECAPA classes are architectures; production needs pretrained weights.
- The IndicF5 wrapper follows the model card's interface and must be verified once installed.
- Joint CTC/attention beam search is not implemented (we rescore an n-best instead).

---

## 11. Glossary

| Term | Meaning |
|---|---|
| AV-HuBERT | Audio-Visual Hidden-unit BERT: self-supervised AV speech model (Meta) |
| AAM-softmax | additive angular margin softmax: cos(θ+m) for the true class |
| AdaBN | adapt BatchNorm statistics to a new domain |
| anusvara / chandrabindu | ं / ँ: nasal marks in Devanagari |
| CER / WER / VER | character / word / viseme error rate |
| CTC | Connectionist Temporal Classification |
| diphone | a pair of adjacent phonemes (a transition) |
| ECAPA-TDNN | speaker-embedding network |
| EER | equal error rate (verification) |
| F0 | fundamental frequency (pitch) |
| flow matching | generative modelling by learning a velocity field from noise to data |
| HiFi-GAN | GAN vocoder |
| LoRA | low-rank adaptation |
| LSGAN | least-squares GAN loss |
| matra | Devanagari vowel sign (ा ि ी …) |
| mel spectrogram | spectrogram on a perceptual frequency scale |
| MOS | mean opinion score (1–5) |
| MPD / MSD | multi-period / multi-scale discriminators |
| nukta | ़: dot changing ज→ज़, फ→फ़ |
| phrase mode | closed-set recognition over the user's phrases |
| schwa deletion | dropping the inherent "a" of Devanagari consonants in speech |
| shallow fusion | adding an LM's log-prob during decoding |
| SyncNet | network that scores audio-video synchrony |
| unit (speech unit) | k-means cluster ID of SSL speech features, 50 Hz |
| viseme | group of phonemes that look the same on the lips |
| virama / halant | ्: removes the inherent vowel |
| VAD | voice activity detection |
| Witten-Bell | an n-gram smoothing method |

---

## 12. Reading list (in priority order)

1. Shi et al., *Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction* (AV-HuBERT), ICLR 2022.
2. Hsu et al., *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units*, 2021.
3. Graves et al., *Connectionist Temporal Classification*, ICML 2006; Hannun, *Sequence Modeling with CTC* (distill.pub, 2017), the best visual explanation.
4. Watanabe et al., *Hybrid CTC/Attention Architecture for End-to-End Speech Recognition*, IEEE JSTSP 2017.
5. Choi, Kim & Ro, *Intelligible Lip-to-Speech Synthesis with Speech Units*, Interspeech 2023 (lip2speech-unit).
6. Yemini et al., *LipVoicer: Generating Speech from Silent Videos Guided by Lip Reading*, ICLR 2024.
7. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.
8. Kong et al., *HiFi-GAN*, NeurIPS 2020; Polyak et al., *Speech Resynthesis from Discrete Disentangled Self-Supervised Representations*, Interspeech 2021.
9. Desplanques et al., *ECAPA-TDNN*, Interspeech 2020; Deng et al., *ArcFace*, CVPR 2019.
10. Chung & Zisserman, *Out of Time: Automated Lip Sync in the Wild* (SyncNet), 2016.
11. Chen et al., *F5-TTS* (flow matching TTS), 2024; AI4Bharat IndicF5 model card.
12. Zanon Boito et al., *mHuBERT-147*, Interspeech 2024.
13. de Cheveigné & Kawahara, *YIN*, JASA 2002.
14. Koehn, *Statistical Significance Tests for Machine Translation Evaluation*, EMNLP 2004 (paired bootstrap).
15. Li et al., *Revisiting Batch Normalization for Practical Domain Adaptation* (AdaBN), 2016.
16. Ohala, *Aspects of Hindi Phonology*, 1983 (schwa deletion).

---

## 13. Commit history (what was built, in order)

Run `git log --oneline` to see it. In order:

1. Package restructure, configs/utilities (+ the YAML float fix)
2. Hindi text layer: normalisation, vocabulary, G2P, visemes
3. Preprocessing: lip crop, audio front-end
4. AV-HuBERT encoder + stackable LoRA
5. Path A recogniser, CTC decoding, phrase mode
6. LLM correction, TTS, Path A pipeline
7. Datasets, synthetic world, metrics
8. Trainer + VSR recipes + the personalisation experiment
9. Path B models (units, pitch, HiFi-GAN)
10. Path B training scripts
11. Voice rebuilding
12. Data engine
13. Hybrid guidance
14. Real-time streaming
15. Recording kit (+ the resume fix)
16. Evaluation runner + CLI
17. GPU configs, CI, merge-and-unload
18. Ablation script, documentation and this guide

Each commit message explains *what* and *why*. Reading them in order is a good way to rebuild the story.

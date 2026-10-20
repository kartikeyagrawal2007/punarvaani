# Lip Lab: see how lip reading performs on your own face, live

A local web dashboard. You mouth phrases at your webcam; it records them, trains two small lip readers on your
laptop, then quizzes you and keeps score. Nothing leaves your computer.

```bash
pip install -e ".[vision]"           # MediaPipe + OpenCV (once)
punarvaani lab --user kartikey        # opens http://127.0.0.1:8765
punarvaani lab --synthetic            # no camera: a cartoon face mouths the phrases for you
```

On a Mac the first run asks for camera access for your terminal app (System Settings → Privacy & Security →
Camera). The MediaPipe face model (3.7 MB) downloads itself to `~/.cache/punarvaani/`.

## How to use it

1. **Record.** The dark card shows a phrase. Keep your lips still for a second (the badge says
   *CALIBRATING*, then *LISTENING*), mouth the phrase, then pause. The take is saved automatically and the
   next phrase appears. Aim for **5 takes per phrase**. If automatic start is unreliable, switch *Start* to
   **Hold Space** and hold the space bar while you mouth.
2. **Train.** Click *Train both readers* (about a minute). You get **held-out accuracy**: every take is
   predicted by a model that never saw it, which is an honest estimate of live accuracy.
3. **Test.** Switch to *Test*. Mouth whatever phrase is prompted; the model is not told which one. You get a
   running score, a per-phrase breakdown and a confusion table (which phrases look alike on your lips).
4. **Free.** Mouth anything from your list and watch what it reads.

Use *Speech: Silent mouthing / Spoken aloud* to tag takes. Comparing the two in Test mode is research
question 2 (the silent-mouthing gap) measured on yourself.

| Flag | Meaning |
|---|---|
| `--user NAME` | takes go to `data/users/NAME/lab/` (git-ignored) |
| `--camera 1` | another webcam |
| `--video clip.mp4` | read a video file instead (looped) |
| `--phrases file.txt` | your own phrase list (one per line); can also be edited in the dashboard |
| `--run outputs/...` | also show the trained open-vocabulary lip reader's output (once Phase 3 training exists) |
| `--port 8765` | change the port |

## What you are looking at

| Panel | Meaning |
|---|---|
| Camera | mirrored webcam; lip outline teal while listening, coral while recording; white dots are the 5 rigid points used to cancel head motion |
| What the model sees | the stabilised 96×96 mouth crop, the same one the real AV-HuBERT model trains on |
| Lip movement chart | how fast the lip shape is changing; recording starts above the coral line and stops after 0.4 s below the teal line |
| Last result | the phrase it read, ✓/✗ against the prompt, and the top-5 of each reader with its confidence |
| Live test score | correct / all attempts; correct when it answered; how often it answered instead of staying silent |

## How it works

```
webcam ─► MediaPipe (478 face points) ─┬─► stabilised mouth crop (96×96)  ─► saved for the real model later
                                       └─► 40 lip points in a canonical face ─► lip-shape features (80 per frame)
                                                    │
                                    lip-activity end-pointer (start / stop)
                                                    │
                                ┌───────────────────┴───────────────────┐
                        Template match (DTW)                 Neural reader (temporal CNN)
                        no training, works from              trained in ~1 min on CPU with
                        1 take per phrase                    augmentation
```

* **Features, not pixels.** The real lip reader reads 88×88 pixel crops and needs hours of data. With 5 takes
  per phrase, the 40 lip landmarks after a similarity transform (rigid points → canonical face) are a far
  smaller input (80 numbers instead of 7,744) that already ignores head position, size and tilt.
  `features.py`.
* **DTW (dynamic time warping).** Lines up the new take against every recorded take, stretching time so a slow
  "पानी" matches a fast one, and picks the closest phrase. The implementation sweeps anti-diagonals so all
  templates are aligned in one vectorised pass (~50 ms for 60 takes). It is the baseline the neural model has
  to beat. `classifiers.py`.
* **Neural reader.** Dilated 1-D convolutions over time (about 1 second of context) with masked mean+max
  pooling. Augmentation (time-stretch 0.8–1.25×, trimming, scale and noise jitter) turns a few takes into
  thousands of variants. The reader with the higher held-out accuracy is the one used live.
* **Rejection.** If the best phrase is below 50% confidence, or the take is far from every recorded phrase,
  the answer is *not sure* and phrase mode stays silent, because a patient would rather say nothing than the
  wrong thing.
* **End-pointing.** The person is silent, so there is no audio to detect speech from. Instead it watches lip
  movement, with separate start and stop thresholds and a 0.4 s hang-over so a short closure (प, ब, म) does
  not cut a phrase in two. The first second is spent measuring your noise floor (`LipActivityDetector` in
  `app/streaming.py`).

## What the numbers mean (and don't)

* This is **phrase mode on your own face**: a closed set of phrases, one speaker. High accuracy here does
  **not** mean open-vocabulary Hindi lip reading works. That is the AV-HuBERT model of Path A, which needs
  the mined Hindi data and GPU training (roadmap phases 2–3). Start the lab with `--run` once it exists to
  see both side by side.
* It *does* tell you, today and on real lips: how separable your phrases are, how many takes are needed,
  how much silent mouthing hurts compared with speaking, and how fast the loop is.
* Every take is saved in the project's manifest format (`train.jsonl` / `test.jsonl` with 96×96 crops), so
  the same recordings can later personalise the real model with `configs/personalize.yaml`.

## Files

| File | What |
|---|---|
| `server.py` | `punarvaani lab`: camera thread, MJPEG streams, JSON API, CLI |
| `session.py` | frames → crop/features → end-pointing → record / predict / score; `TakeStore` on disk |
| `features.py` | canonical lip-shape features |
| `classifiers.py` | batched DTW, temporal-CNN reader, cross-validation |
| `synthetic.py` | cartoon face with landmarks, for `--synthetic` and the tests |
| `static/index.html` | the dashboard (plain HTML/JS, no internet needed) |

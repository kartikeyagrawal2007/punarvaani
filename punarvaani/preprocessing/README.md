# preprocessing

| File | What |
|---|---|
| `lip_crop.py` | MediaPipe Face Mesh landmarks → interpolate gaps → temporal smoothing → similarity alignment on rigid points → 96×96 grayscale mouth crop (88×88 random/centre crop at train/test) |
| `audio.py` | 16 kHz I/O, resampling, log-mel at 100 Hz, frame stacking to 25 Hz, RMS normalisation, noise-adaptive energy VAD; the 25/50/100 Hz frame-rate contract |
| `pitch.py` | YIN F0 tracker at 50 Hz and speaker-normalised pitch targets for Path B |

**Owner:** Vision model lead

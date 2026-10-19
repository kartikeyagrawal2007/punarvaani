# voice_rebuild

Rebuild a user's voice from old voice notes and family videos.

| File | What |
|---|---|
| `denoise.py` | DeepFilterNet wrapper, spectral Wiener fallback, quality report (percentile SNR, clipping, bandwidth) |
| `speaker.py` | ECAPA-TDNN speaker encoder and AAM-softmax loss |
| `rebuild.py` | segment → score → embed → cluster speakers → keep the patient → speaker embedding, ~10 s reference clip, clean segments |

Run: `punarvaani rebuild-voice notes/*.wav --out profiles/P07 --enroll notes/known_P07.wav --ecapa ecapa.pt`

**Owner:** Speech model lead

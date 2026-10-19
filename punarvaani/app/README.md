# app

Real time: phone or webcam → laptop/edge GPU → audio back. Target under 1 second end to end.

| File | What |
|---|---|
| `streaming.py` | causal online lip cropper, hysteresis lip-activity end-pointer, utterance streamer (Path A), chunked unit streamer (Path B) |
| `realtime.py` | webcam live session: MediaPipe → crop → end-pointing → Path A → speaker |

**Owner:** System lead

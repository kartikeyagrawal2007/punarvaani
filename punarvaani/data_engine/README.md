# data_engine

Hindi audio-visual mining: video URL → frames + audio → ASR word timestamps → clips at pauses → text filters → single-shot check → lip crop → A/V sync check → `crops/*.npy`, `wav/*.wav`, `train/val/test.jsonl` (speaker-disjoint). Target 50–100 h. Output goes to the shared Drive, never to git.

| File | What |
|---|---|
| `segment.py` | word timestamps → 1–12 s clips that start and end in pauses |
| `sync.py` | SyncNet offset/confidence scoring, a small two-stream SyncNet, InfoNCE loss |
| `tracking.py` | shot-cut detection, IoU face tracker, usable-track filters |
| `pipeline.py` | `MiningPipeline` with a per-reason `FilterReport`, speaker-disjoint splits, yt-dlp/ffmpeg/ASR helpers |

Run: `punarvaani mine --urls urls.txt --out data/hindi_av --face-model face_landmarker.task`

**Owner:** Data lead

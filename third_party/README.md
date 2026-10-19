# third_party

We re-implemented the parts we need (AV-HuBERT encoder, HiFi-GAN, ECAPA, SyncNet scoring) in `punarvaani/`, so no upstream code is vendored here. What we *do* need from upstream is **weights**:

| Weights | Source | Used by |
|---|---|---|
| AV-HuBERT Large (`large_vox_iter5.pt`, self-supervised) | github.com/facebookresearch/av_hubert | `model.avhubert_checkpoint` |
| mHuBERT-147 | Hugging Face `utter-project/mHuBERT-147` | `HuBERTUnitExtractor` |
| IndicF5 | Hugging Face `ai4bharat/IndicF5` | `IndicF5TTS` |
| MediaPipe face landmarker | storage.googleapis.com/mediapipe-models/face_landmarker/... | `MediaPipeLandmarker` |
| SyncNet v2, ECAPA (SpeechBrain) | respective repos | data filtering, voice rebuild |

Upstream repos can still be added here as git submodules for reference. Check each license: AV-HuBERT is non-commercial research only.

# models/path_b

Path B ("ventriloquist"): lips → 50 Hz speech units (+ mel, + pitch) → unit HiFi-GAN conditioned on the user's speaker embedding.

| File | What |
|---|---|
| `model.py` | `LipToUnitModel`: unit, mel, pitch and auxiliary CTC heads on the shared encoder |
| `units.py` | torch k-means (k-means++), mHuBERT-147 unit extractor, run-length dedup |
| `vocoder.py` | unit HiFi-GAN generator, multi-period + multi-scale discriminators, GAN/feature/mel losses |

**Owner:** Speech model lead

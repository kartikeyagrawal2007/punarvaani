# models/encoder

| File | What |
|---|---|
| `resnet.py` | 3D-conv stem + ResNet-18 trunk (AV-HuBERT's visual front-end, fairseq names) |
| `transformer.py` | wav2vec2-style transformer: conv positional embedding, separate q/k/v projections, pre/post-LN, LayerDrop |
| `avhubert.py` | `VisualSpeechEncoder` (presets tiny/small/base/large), time masking, fairseq checkpoint conversion/loading |
| `lora.py` | named, stackable LoRA adapters (`hindi` + `user_<id>`), merge/unmerge, `merge_and_unload`, per-adapter save/load |

**Owner:** Vision model lead

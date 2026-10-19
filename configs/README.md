# configs

YAML configs. A `_base_: other.yaml` key inherits another config; any key can be overridden on the command line (`train.lr=3e-4`).

| Config | What it trains | Hardware |
|---|---|---|
| `synthetic_vsr.yaml` | tiny Path A model on the synthetic lip world | CPU, ~1.5 min |
| `synthetic_personalize.yaml` | per-user LoRA on a "silent" synthetic user | CPU, ~30 s |
| `synthetic_units.yaml` | tiny Path B lips → units | CPU, ~1 min |
| `synthetic_vocoder.yaml` | tiny unit HiFi-GAN | CPU, ~1 min |
| `hindi_adapt.yaml` | AV-HuBERT-Large + `hindi` LoRA on mined Hindi data | 1 GPU (A100, or T4 with smaller batches) |
| `personalize.yaml` | `user_<id>` adapter on a user's silent mouthing | 1 GPU, < 1 h |
| `path_b_units.yaml` | Path B units/mel heads on the Hindi encoder | 1 GPU |
| `vocoder_pretrain.yaml` / `vocoder_finetune.yaml` | multi-speaker vocoder, then one patient's voice | 1 GPU |
| `phrases_hi.txt` | starter phrase bank for phrase mode | — |

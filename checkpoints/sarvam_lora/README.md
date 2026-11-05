# Sarvam-1 LoRA Adapter Drop Folder

Put the exported PEFT/LoRA adapter files from the Stage-2 owner directly in this folder:

```text
checkpoints/sarvam_lora/
├── adapter_config.json
├── adapter_model.safetensors      (or adapter_model.bin)
├── tokenizer_config.json
├── special_tokens_map.json
└── tokenizer.json                 (if exported)
```

The adapter must be trained from `sarvamai/sarvam-1` with the four special tokens:
`<S2S>`, `<PHONEMES>`, `</PHONEMES>`, `<TEXT>`.

Validate it before training/evaluation:

```bash
python scripts/preflight.py --data_dir ./hindi_dataset --sarvam_adapter_path ./checkpoints/sarvam_lora
```

or directly:

```bash
python -c "from Models.Sarvam import validate_sarvam_adapter_compatibility as v; print(v('checkpoints/sarvam_lora')); assert v('checkpoints/sarvam_lora')['is_compatible']"
```

If the adapter is missing or incompatible, the pipeline falls back to **MOCK mode** with an
explicit reason instead of silently using the wrong model. Adapter weights are gitignored and
must never be uploaded publicly.

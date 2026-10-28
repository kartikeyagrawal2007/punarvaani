# Fine-Tuned Sarvam-1 LoRA Checkpoint Directory

Drop the exported PEFT/LoRA adapter files from your teammate directly into this directory:

```text
checkpoints/sarvam_lora/
├── adapter_config.json
├── adapter_model.safetensors (or adapter_model.bin)
├── special_tokens_map.json
├── tokenizer_config.json
└── tokenizer.json
```

### Notes:
- When running inference with `--sarvam_adapter_path ./checkpoints/sarvam_lora`, the model loads these weights.
- If this directory is empty or the flag is omitted, the pipeline runs in MOCK mode without crashing.

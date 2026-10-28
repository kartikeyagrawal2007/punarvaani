"""
Sarvam-1 Connector for Hindi Visual Speech Recognition (VSR)
Connects the predicted CTC phoneme sequence to the fine-tuned Sarvam-1 Indic LLM
to reconstruct coherent Devanagari Hindi text.
"""

import os
import torch
from typing import List, Optional

TAGS = ["<S2S>", "<PHONEMES>", "</PHONEMES>", "<TEXT>"]


class HindiPhonemeTranslator:
    """
    Bridge connecting Stage 1 (Video-to-Phoneme CTC) output to Stage 2 (Sarvam-1 Indic LLM).
    Operates in MOCK mode when adapter weights are not yet available, allowing
    the entire pipeline to be developed, tested, and validated without waiting for LLM training.
    """
    def __init__(
        self,
        adapter_path: Optional[str] = None,
        base_model_id: str = "sarvamai/sarvam-1",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.adapter_path = adapter_path
        self.base_model_id = base_model_id
        self.device = device
        self.model = None
        self.tokenizer = None
        self.is_mock = True

        if adapter_path and os.path.exists(adapter_path):
            self._load_fine_tuned_model()
        else:
            if adapter_path:
                print(f"[HindiPhonemeTranslator] Adapter path '{adapter_path}' not found. Falling back to MOCK mode.")
            else:
                print("[HindiPhonemeTranslator] Initialized in MOCK mode (no adapter path specified).")

    def _load_fine_tuned_model(self):
        """Loads base Sarvam-1 model and mounts PEFT/LoRA adapters."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            print(f"[HindiPhonemeTranslator] Loading base tokenizer and model from '{self.base_model_id}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, use_fast=True)
            self.tokenizer.add_special_tokens({"additional_special_tokens": TAGS})
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                device_map="auto" if "cuda" in self.device else None
            )
            base_model.resize_token_embeddings(len(self.tokenizer))

            print(f"[HindiPhonemeTranslator] Mounting LoRA adapter from '{self.adapter_path}'...")
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            self.model.eval()
            self.is_mock = False
            print("[HindiPhonemeTranslator] Successfully loaded fine-tuned Sarvam-1 model!")

        except Exception as e:
            print(f"[HindiPhonemeTranslator] Warning: Failed to load Sarvam model ({e}). Reverting to MOCK mode.")
            self.is_mock = True

    def build_prompt(self, phoneme_sequence: str) -> str:
        """Formats the input prompt using the shared S2S phoneme contract."""
        clean_phonemes = phoneme_sequence.strip()
        prompt = (
            f"{TAGS[0]}\n"
            f"{TAGS[1]}\n{clean_phonemes}\n{TAGS[2]}\n"
            f"{TAGS[3]}\n"
        )
        return prompt

    def translate(self, phoneme_sequence: str, max_new_tokens: int = 64) -> str:
        """
        Translates a predicted phoneme string (e.g. 'N M S TD E') into Devanagari Hindi text.
        """
        if not phoneme_sequence or not phoneme_sequence.strip():
            return ""

        if self.is_mock or self.model is None:
            # MOCK mode: Return a clean placeholder showing what phonemes were received
            return f"[MOCK TRANSCRIPTION: {phoneme_sequence}]"

        prompt = self.build_prompt(phoneme_sequence)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Slice off prompt tokens to extract only generated Hindi response
        prompt_len = inputs.input_ids.shape[1]
        generated_ids = outputs[0][prompt_len:]
        decoded_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Stop at first newline if present
        first_line = decoded_text.split("\n")[0].strip()
        return first_line

    def batch_translate(self, phoneme_sequences: List[str], max_new_tokens: int = 64) -> List[str]:
        return [self.translate(p, max_new_tokens=max_new_tokens) for p in phoneme_sequences]


# Standalone training script template for teammate fine-tuning Sarvam-1
def train_sarvam_lora(
    train_jsonl_path: str,
    val_jsonl_path: str,
    output_dir: str = "./sarvam_phoneme_to_text_lora",
    base_model_id: str = "sarvamai/sarvam-1",
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 2e-4
):
    """
    Utility function for the teammate fine-tuning Sarvam-1.
    Expects JSONL files with rows: {"prompt": "<S2S>\\n...", "target": "नमस्ते"}
    """
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": TAGS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files={"train": train_jsonl_path, "val": val_jsonl_path})

    def tokenize_fn(batch):
        prompts = [p + t for p, t in zip(batch["prompt"], batch["target"])]
        enc = tokenizer(prompts, max_length=512, truncation=True, padding="max_length")
        labels = []
        for p, t in zip(batch["prompt"], batch["target"]):
            p_len = len(tokenizer(p, add_special_tokens=False)["input_ids"])
            total_ids = tokenizer(p + t, max_length=512, truncation=True)["input_ids"]
            lab = [-100] * p_len + total_ids[p_len:]
            lab = lab + [-100] * (512 - len(lab))
            labels.append(lab[:512])
        enc["labels"] = labels
        return enc

    tokenized = dataset.map(tokenize_fn, batched=True)

    model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16, device_map="auto")
    model.resize_token_embeddings(len(tokenizer))

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_cfg)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        fp16=True,
        save_total_limit=2,
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned Sarvam-1 adapter saved to {output_dir}")

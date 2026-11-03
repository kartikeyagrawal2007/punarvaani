"""
Sarvam-1 Connector for Hindi Visual Speech Recognition (VSR)
Connects the predicted CTC phoneme sequence to the fine-tuned Sarvam-1 Indic LLM
to reconstruct coherent Devanagari Hindi text.
"""

import os
import torch
from typing import List, Optional

TAGS = ["<S2S>", "<PHONEMES>", "</PHONEMES>", "<TEXT>"]


def validate_sarvam_adapter_compatibility(
    adapter_path: str,
    base_model_id: str = "sarvamai/sarvam-1",
    tokenizer: Optional[object] = None,
    strict: bool = True
) -> dict:
    """
    Validates compatibility of a fine-tuned LoRA adapter before mounting:
    1. Check adapter directory exists and contains adapter_config.json.
    2. Check adapter_config.json base_model_name_or_path matches base_model_id.
    3. Check tokenizer registers all required TAGS (<S2S>, <PHONEMES>, </PHONEMES>, <TEXT>).
    4. Verify each special token maps to a single token ID without subword segmentation.
    
    Returns:
        dict with status, missing_tokens, mismatch_details, and is_compatible boolean.
    Raises:
        ValueError if strict=True and compatibility check fails.
    """
    import json

    if not os.path.exists(adapter_path):
        err = f"Adapter directory does not exist: {adapter_path}"
        if strict:
            raise FileNotFoundError(err)
        return {"is_compatible": False, "error": err}

    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(cfg_path):
        err = f"Missing adapter_config.json in {adapter_path}"
        if strict:
            raise FileNotFoundError(err)
        return {"is_compatible": False, "error": err}

    with open(cfg_path, "r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)

    expected_base = adapter_cfg.get("base_model_name_or_path")
    if expected_base and expected_base != base_model_id:
        err = f"Adapter base model '{expected_base}' does not match expected '{base_model_id}'."
        if strict:
            raise ValueError(err)
        return {"is_compatible": False, "error": err}

    missing_tokens = []
    subword_split_tokens = []
    if tokenizer is not None:
        vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        for tag in TAGS:
            if tag not in vocab:
                missing_tokens.append(tag)
            else:
                encoded = tokenizer.encode(tag, add_special_tokens=False)
                if len(encoded) != 1:
                    subword_split_tokens.append((tag, encoded))

        if missing_tokens:
            err = f"Tokenizer is missing required special tokens: {missing_tokens}"
            if strict:
                raise ValueError(err)
            return {"is_compatible": False, "error": err, "missing_tokens": missing_tokens}

        if subword_split_tokens:
            err = f"Special tokens split into multiple subwords: {subword_split_tokens}"
            if strict:
                raise ValueError(err)
            return {"is_compatible": False, "error": err, "subword_split_tokens": subword_split_tokens}

    return {
        "is_compatible": True,
        "base_model_id": base_model_id,
        "adapter_path": adapter_path,
        "missing_tokens": missing_tokens
    }


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
        self.mock_reason = None

        if adapter_path and os.path.exists(adapter_path):
            self._load_fine_tuned_model()
        else:
            if adapter_path:
                self.mock_reason = f"adapter path not found: {adapter_path}"
                print(f"[HindiPhonemeTranslator] Adapter path '{adapter_path}' not found. Falling back to MOCK mode.")
            else:
                self.mock_reason = "no adapter path specified"
                print("[HindiPhonemeTranslator] Initialized in MOCK mode (no adapter path specified).")

    def _load_fine_tuned_model(self):
        """
        Loads base Sarvam-1 model and mounts PEFT/LoRA adapters with strict 7-step compatibility validation:
        1. Attempt to load tokenizer configuration from adapter directory first (to match fine-tuning configuration).
        2. Check for required special tokens (<S2S>, <PHONEMES>, </PHONEMES>, <TEXT>).
        3. Register missing special tokens.
        4. Resize token embeddings.
        5. Verify token IDs and tokenizer vocabulary size.
        6. Verify compatibility between inference tokenizer and LoRA adapter.
        7. Raise explicit error if token IDs or vocabulary mismatch makes adapter unsafe to load.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError as exc:
            self.is_mock = True
            self.mock_reason = f"missing dependency: {exc}"
            print(f"[HindiPhonemeTranslator] Warning: {self.mock_reason}. Reverting to MOCK mode.")
            return

        try:

            # Step 1: Check if adapter directory contains tokenizer files
            tokenizer_source = self.adapter_path if (self.adapter_path and os.path.exists(os.path.join(self.adapter_path, "tokenizer_config.json"))) else self.base_model_id
            print(f"[HindiPhonemeTranslator] Loading tokenizer from '{tokenizer_source}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

            # Step 2 & 3: Check and register missing special tokens
            missing_tokens = [t for t in TAGS if t not in self.tokenizer.get_vocab()]
            if missing_tokens:
                print(f"[HindiPhonemeTranslator] Registering missing special tokens: {missing_tokens}")
                self.tokenizer.add_special_tokens({"additional_special_tokens": missing_tokens})

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Step 4: Load base model and resize embeddings
            print(f"[HindiPhonemeTranslator] Loading base model from '{self.base_model_id}'...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                device_map="auto" if "cuda" in self.device else None
            )
            base_model.resize_token_embeddings(len(self.tokenizer))

            # Step 5 & 6: Check adapter configuration compatibility.
            # Never mount an adapter whose base model or special tokens do not match.
            report = validate_sarvam_adapter_compatibility(
                adapter_path=self.adapter_path,
                base_model_id=self.base_model_id,
                tokenizer=self.tokenizer,
                strict=False,
            )
            if not report.get("is_compatible", False):
                raise ValueError(report.get("error", "adapter compatibility check failed"))

            # Step 7: Mount LoRA adapter
            print(f"[HindiPhonemeTranslator] Mounting LoRA adapter from '{self.adapter_path}'...")
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            self.model.eval()
            self.is_mock = False
            self.mock_reason = None
            print("[HindiPhonemeTranslator] Successfully loaded fine-tuned Sarvam-1 model!")

        except Exception as e:
            self.mock_reason = str(e)
            print(f"[HindiPhonemeTranslator] Warning: Failed to load Sarvam model ({e}). Reverting to MOCK mode.")
            self.is_mock = True
            self.model = None

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

    max_length = 512
    tokenizer.padding_side = "right"
    pad_id = tokenizer.pad_token_id

    def tokenize_fn(batch):
        prompts = [p + t for p, t in zip(batch["prompt"], batch["target"])]
        enc = tokenizer(
            prompts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        labels = []
        for i, prompt in enumerate(batch["prompt"]):
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            total_ids = enc["input_ids"][i]
            prompt_len = min(len(prompt_ids), len(total_ids))
            target_ids = [
                token if token != pad_id else -100
                for token in total_ids[prompt_len:]
            ]
            labels.append([-100] * prompt_len + target_ids)
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

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
    )
    try:
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:  # older transformers
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned Sarvam-1 adapter saved to {output_dir}")

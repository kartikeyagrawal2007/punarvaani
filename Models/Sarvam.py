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
            print(f"[HindiPhonemeTranslator] Loading tokenizer from '{tokenizer_source}'...", flush=True)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
            except Exception as exc:
                if tokenizer_source == self.base_model_id:
                    raise
                # tokenizer.save_pretrained() on a fast Llama tokenizer can omit
                # tokenizer.model while tokenizer_config.json still says
                # LlamaTokenizer. Fall back to the base tokenizer; the missing
                # tags are added below and keep the same token IDs.
                print(
                    f"[HindiPhonemeTranslator] Adapter tokenizer failed ({exc}); "
                    "falling back to the base Sarvam tokenizer.",
                    flush=True,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, use_fast=True)

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
    lr: float = 2e-4,
    max_length: int = 512,
    gradient_accumulation_steps: int = 1,
    eval_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    bf16: Optional[bool] = None,
):
    """
    Utility function for the teammate fine-tuning Sarvam-1.
    Expects JSONL files with rows: {"prompt": "<S2S>\\n...", "target": "नमस्ते"}.

    The four tags (<S2S>, <PHONEMES>, </PHONEMES>, <TEXT>) are new tokens. Their
    embedding rows are included in the saved adapter via modules_to_save so the
    repo's HindiPhonemeTranslator sees the same vectors used during training.
    """
    import inspect

    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

    # bf16 does not use GradScaler. If the GPU cannot do bf16, load the model
    # in fp32 and let fp16 autocast + GradScaler handle the mixed precision.
    if bf16 is None:
        bf16 = bool(
            torch.cuda.is_available()
            and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
    use_fp16 = bool(torch.cuda.is_available() and not bf16)
    print(f"[Sarvam] precision: {'bf16' if bf16 else ('fp16' if use_fp16 else 'fp32')}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": TAGS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = load_dataset("json", data_files={"train": train_jsonl_path, "val": val_jsonl_path})

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    stats = {
        "empty_target": 0,
        "truncated": 0,
        "prompt_too_long": 0,
        "bad_prompt": 0,
        "crlf": 0,
    }

    def tokenize_fn(batch):
        input_ids, attention_mask, labels = [], [], []
        for prompt, target in zip(batch["prompt"], batch["target"]):
            if not target.strip():
                stats["empty_target"] += 1
            if not all(tag in prompt for tag in TAGS):
                stats["bad_prompt"] += 1
            if "\r" in prompt:
                stats["crlf"] += 1

            # The tokenizer adds BOS to the prompt (add_bos_token=True) but not
            # EOS to the target (add_eos_token=False), so EOS is appended here.
            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            target_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + [eos_id]

            if len(prompt_ids) >= max_length:
                stats["prompt_too_long"] += 1

            ids = prompt_ids + target_ids
            lab = [-100] * len(prompt_ids) + target_ids

            if len(ids) > max_length:
                stats["truncated"] += 1
                ids = ids[:max_length]
                lab = lab[:max_length]

            pad = max_length - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
            labels.append(lab + [-100] * pad)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    print(f"[Sarvam] tokenization stats: {stats}")
    if stats["empty_target"] or stats["bad_prompt"] or stats["crlf"]:
        print(
            "[Sarvam] WARNING: empty targets, prompts missing tags, or CRLF characters found. "
            "Fix the JSONL before trusting the adapter."
        )
    if stats["truncated"] or stats["prompt_too_long"]:
        print(
            f"[Sarvam] WARNING: some samples exceed max_length={max_length}. "
            "Targets may be truncated; check the prompt/target length distribution."
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
    )
    if torch.cuda.is_available():
        model.to("cuda")
    model.config.use_cache = False
    model.resize_token_embeddings(len(tokenizer))

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    print(
        "[Sarvam] embed_tokens/lm_head are saved with the adapter so the four new tags "
        "survive save/load; expect a roughly 500-600 MB adapter."
    )

    # Transformers renamed evaluation_strategy -> eval_strategy. Pick the
    # argument the installed version actually accepts, otherwise
    # load_best_model_at_end sees evaluation_strategy="no" and raises ValueError.
    training_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=lr,
        fp16=use_fp16,
        bf16=bf16,
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=5,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        report_to="none",
    )
    training_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_arg = "eval_strategy" if "eval_strategy" in training_params else "evaluation_strategy"
    training_kwargs[eval_arg] = "steps"
    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
    )
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
    else:
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[Sarvam] Fine-tuned Sarvam-1 adapter saved to {output_dir}")
    print(
        "[Sarvam] Check that adapter_model.safetensors is roughly 500-600 MB. "
        "If it is only 20-40 MB, the tag embeddings were not saved; do not paste it."
    )
    return trainer

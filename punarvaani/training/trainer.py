"""A small, explicit training loop shared by every model in the project.

Features that matter for our setting (one Colab/Kaggle GPU, fine-tuning big
pretrained models on little data):

* mixed precision (bf16 where supported, fp16 + GradScaler otherwise);
* gradient accumulation, so an effective batch of many clips fits in 16 GB;
* gradient clipping (transformers + CTC occasionally spike);
* linear warm-up then cosine decay - warm-up protects pretrained weights from
  large early updates while the new heads are still random;
* **frozen BatchNorm**: a frozen ResNet in ``train()`` mode would still update
  its BN running statistics from our small batches and silently drift away
  from the pretrained statistics. We put BN layers with frozen parameters in
  eval mode after every ``model.train()``;
* checkpoint/resume (Colab sessions die) and "keep the best by validation metric".
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from punarvaani.utils import get_logger

log = get_logger(__name__)


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.98)
    warmup_steps: int = 100
    max_steps: int = 1000
    min_lr_ratio: float = 0.05
    grad_accum: int = 1
    clip_norm: float = 5.0
    amp: str = "auto"  # "auto" | "bf16" | "fp16" | "none"
    log_every: int = 20
    eval_every: int = 200
    out_dir: str = "outputs/run"
    save_trainable_only: bool = False
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict | None) -> TrainConfig:
        d = dict(d or {})
        if "betas" in d:
            d["betas"] = tuple(d["betas"])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """No weight decay on biases, norms, embeddings and LoRA ``B`` matrices."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "norm" in name or "embed" in name or "mask_emb" in name or "lora_B" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def warmup_cosine(step: int, warmup: int, total: int, min_ratio: float) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def freeze_frozen_batchnorm(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and not any(
            p.requires_grad for p in m.parameters()
        ):
            m.eval()


def _amp_dtype(mode: str, device: torch.device) -> torch.dtype | None:
    if mode == "none" or device.type != "cuda":
        return None
    if mode == "bf16" or (mode == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16


def move_to(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_to(v, device) for k, v in batch.items()}
    return batch


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[[nn.Module, dict], tuple[torch.Tensor, dict]],
        train_loader: Iterable,
        cfg: TrainConfig,
        eval_fn: Callable[[nn.Module], dict] | None = None,
        metric: str = "cer",
        device: torch.device | None = None,
    ):
        self.model, self.loss_fn, self.train_loader, self.cfg = model, loss_fn, train_loader, cfg
        self.eval_fn, self.metric = eval_fn, metric
        self.device = device or next(model.parameters()).device
        self.optimizer = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=cfg.betas)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda s: warmup_cosine(s, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio)
        )
        self.amp_dtype = _amp_dtype(cfg.amp, self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_dtype == torch.float16)
        self.step, self.best = 0, math.inf
        self.out_dir = Path(cfg.out_dir)
        self.history: list[dict] = []

    # -- checkpoints ---------------------------------------------------------
    def _model_state(self) -> dict:
        state = self.model.state_dict()
        if self.cfg.save_trainable_only:
            trainable = {n for n, p in self.model.named_parameters() if p.requires_grad}
            state = {k: v for k, v in state.items() if k in trainable}
        return state

    def save(self, name: str) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / name
        torch.save({"model": self._model_state(), "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(), "step": self.step, "best": self.best,
                    "config": asdict(self.cfg)}, path)  # fmt: skip
        return path

    def resume(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"], strict=not self.cfg.save_trainable_only)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.step, self.best = ckpt["step"], ckpt["best"]
        log.info("resumed from %s at step %d", path, self.step)

    # -- loop ----------------------------------------------------------------
    def _train_mode(self) -> None:
        self.model.train()
        freeze_frozen_batchnorm(self.model)

    def evaluate(self) -> dict:
        if self.eval_fn is None:
            return {}
        self.model.eval()
        with torch.no_grad():
            metrics = self.eval_fn(self.model)
        self._train_mode()
        value = metrics.get(self.metric)
        if value is not None and value < self.best:
            self.best = value
            self.save("best.pt")
        return metrics

    def fit(self) -> list[dict]:
        cfg = self.cfg
        self._train_mode()
        t0, running, n_running = time.time(), {}, 0
        data_iter = self._infinite(self.train_loader)
        while self.step < cfg.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            for _ in range(cfg.grad_accum):
                batch = move_to(next(data_iter), self.device)
                with torch.autocast(
                    self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None
                ):
                    loss, logs = self.loss_fn(self.model, batch)
                self.scaler.scale(loss / cfg.grad_accum).backward()
                for k, v in logs.items():
                    running[k] = running.get(k, 0.0) + float(v)
                n_running += 1
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], cfg.clip_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.step += 1

            if self.step % cfg.log_every == 0 or self.step == cfg.max_steps:
                entry = {"step": self.step, "lr": self.scheduler.get_last_lr()[0], "grad_norm": float(grad_norm),
                         **{k: v / n_running for k, v in running.items()}, "elapsed_s": round(time.time() - t0, 1)}  # fmt: skip
                log.info(
                    " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in entry.items())
                )
                self.history.append(entry)
                running, n_running = {}, 0
            if self.eval_fn is not None and (self.step % cfg.eval_every == 0 or self.step == cfg.max_steps):
                metrics = self.evaluate()
                log.info("eval step=%d %s (best %s=%.4f)", self.step, metrics, self.metric, self.best)
                self.history.append({"step": self.step, "eval": metrics})
        self.save("last.pt")
        (self.out_dir / "history.json").write_text(json.dumps(self.history, ensure_ascii=False, indent=1))
        return self.history

    @staticmethod
    def _infinite(loader):
        epoch = 0
        while True:
            sampler = getattr(loader, "batch_sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            yield from loader
            epoch += 1

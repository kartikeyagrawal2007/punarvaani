"""Small helpers shared by every module: seeding, devices, padding masks."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_logger(name: str = "punarvaani") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def lengths_to_padding_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """Boolean mask of shape (B, T) that is True at *padded* positions.

    This is the convention ``nn.MultiheadAttention(key_padding_mask=...)`` uses.
    """
    max_len = int(max_len if max_len is not None else lengths.max().item())
    steps = torch.arange(max_len, device=lengths.device)
    return steps.unsqueeze(0) >= lengths.unsqueeze(1)


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)

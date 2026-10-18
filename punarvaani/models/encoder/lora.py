"""Low-Rank Adaptation (LoRA, Hu et al. 2021) with *named, stackable* adapters.

Fine-tuning all 300M+ parameters of AV-HuBERT-Large on 50-100 h of Hindi would
overfit and needs a lot of GPU memory (weights + gradients + Adam states ≈ 4x
the model). LoRA freezes the pretrained weight ``W`` and learns a low-rank
update::

    y = W x + (alpha / r) * B (A x),     A: r x d_in,  B: d_out x r,  r << d

``B`` starts at zero, so training starts exactly from the pretrained model.
With r = 16 on q/v/out/ffn of a 1024-d model this is ~1-2% of the parameters.

Why our own implementation instead of ``peft``? We need two adapters at once:

* ``hindi`` - learned once on the mined Hindi dataset (language adaptation);
* ``user_<id>`` - learned per patient on 10-15 min of *silent mouthing*
  (speaker + silent-articulation adaptation), applied on top of ``hindi``.

One frozen base serves every patient; a patient's adapter is a few MB and can be
swapped at runtime. The ``hindi`` adapter can also be merged into ``W`` for
zero-overhead inference. Writing it ourselves keeps that logic explicit (and is
~150 lines).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch
from torch import nn

DEFAULT_TARGETS = ("q_proj", "v_proj", "out_proj", "fc1", "fc2")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self.scales: dict[str, float] = {}
        self.dropouts = nn.ModuleDict()
        self.active: list[str] = []
        self.merged: set[str] = set()
        # A freshly built module defaults to training mode; inherit the wrapped
        # layer's mode so injecting into an eval() model doesn't switch on dropout.
        self.train(base.training)

    in_features = property(lambda self: self.base.in_features)
    out_features = property(lambda self: self.base.out_features)
    weight = property(lambda self: self.base.weight)
    bias = property(lambda self: self.base.bias)

    def add_adapter(self, name: str, r: int, alpha: float, dropout: float = 0.0) -> None:
        if name in self.lora_A:
            raise ValueError(f"adapter {name!r} already exists")
        a = nn.Parameter(torch.empty(r, self.in_features, device=self.weight.device, dtype=self.weight.dtype))
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        b = nn.Parameter(
            torch.zeros(self.out_features, r, device=self.weight.device, dtype=self.weight.dtype)
        )
        self.lora_A[name], self.lora_B[name] = a, b
        self.scales[name] = alpha / r
        self.dropouts[name] = (nn.Dropout(dropout) if dropout > 0 else nn.Identity()).train(self.training)
        self.active.append(name)

    def delta_weight(self, name: str) -> torch.Tensor:
        return (self.lora_B[name] @ self.lora_A[name]) * self.scales[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        for name in self.active:
            if name in self.merged:
                continue
            y = y + (self.dropouts[name](x) @ self.lora_A[name].T @ self.lora_B[name].T) * self.scales[name]
        return y

    @torch.no_grad()
    def merge(self, name: str) -> None:
        if name not in self.merged:
            self.base.weight += self.delta_weight(name)
            self.merged.add(name)

    @torch.no_grad()
    def unmerge(self, name: str) -> None:
        if name in self.merged:
            self.base.weight -= self.delta_weight(name)
            self.merged.discard(name)


def _lora_modules(model: nn.Module) -> Iterable[tuple[str, LoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def inject_lora(
    model: nn.Module,
    adapter: str,
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    targets: Sequence[str] = DEFAULT_TARGETS,
    layers: Sequence[int] | None = None,
) -> int:
    """Wrap every ``nn.Linear`` whose attribute name is in ``targets`` and add ``adapter``.

    ``layers`` restricts injection to those transformer layer indices (matched
    on ``.layers.<i>.`` in the module path), e.g. only the top half. Returns the
    number of adapted linears.
    """
    count = 0
    for parent_name, parent in list(model.named_modules()):
        if layers is not None and not any(f"layers.{i}." in parent_name + "." for i in layers):
            continue
        for child_name, child in list(parent.named_children()):
            if child_name not in targets:
                continue
            if isinstance(child, nn.Linear):
                child = LoRALinear(child)
                setattr(parent, child_name, child)
            if isinstance(child, LoRALinear) and adapter not in child.lora_A:
                child.add_adapter(adapter, r, alpha, dropout)
                count += 1
    if count == 0:
        raise ValueError(f"no linear layer named any of {targets} was found")
    return count


def mark_trainable(model: nn.Module, adapters: Sequence[str], extra: Sequence[str] = ()) -> int:
    """Freeze everything except the given adapters and parameters whose name contains any of ``extra``.

    Returns the number of trainable parameters.
    """
    for name, p in model.named_parameters():
        is_adapter = any(f"lora_A.{a}" in name or f"lora_B.{a}" in name for a in adapters)
        p.requires_grad = is_adapter or any(e in name for e in extra)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_active_adapters(model: nn.Module, adapters: Sequence[str]) -> None:
    for _, m in _lora_modules(model):
        m.active = [a for a in adapters if a in m.lora_A]


def merge_adapter(model: nn.Module, adapter: str) -> None:
    for _, m in _lora_modules(model):
        if adapter in m.lora_A:
            m.merge(adapter)


def unmerge_adapter(model: nn.Module, adapter: str) -> None:
    for _, m in _lora_modules(model):
        if adapter in m.lora_A:
            m.unmerge(adapter)


def adapter_state_dict(model: nn.Module, adapter: str) -> dict[str, torch.Tensor]:
    state = {}
    for name, p in model.named_parameters():
        if name.endswith(f"lora_A.{adapter}") or name.endswith(f"lora_B.{adapter}"):
            state[name] = p.detach().cpu().clone()
    return state


def save_adapter(model: nn.Module, adapter: str, path: str | Path, meta: dict | None = None) -> None:
    scales = {n: m.scales[adapter] for n, m in _lora_modules(model) if adapter in m.scales}
    torch.save(
        {
            "adapter": adapter,
            "state": adapter_state_dict(model, adapter),
            "scales": scales,
            "meta": meta or {},
        },
        path,
    )


def load_adapter(model: nn.Module, path: str | Path, activate: bool = True) -> str:
    """Load an adapter saved by :func:`save_adapter`, creating its LoRA slots as needed."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    name, state = blob["adapter"], blob["state"]
    modules = dict(_lora_modules(model))
    for module_name, scale in blob["scales"].items():
        a = state[f"{module_name}.lora_A.{name}"]
        if module_name not in modules:
            parent_name, _, child = module_name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            modules[module_name] = LoRALinear(getattr(parent, child))
            setattr(parent, child, modules[module_name])
        m = modules[module_name]
        if name not in m.lora_A:
            m.add_adapter(name, r=a.shape[0], alpha=scale * a.shape[0])
        m.lora_A[name].data.copy_(a)
        m.lora_B[name].data.copy_(state[f"{module_name}.lora_B.{name}"])
        if not activate:
            m.active.remove(name)
    return name

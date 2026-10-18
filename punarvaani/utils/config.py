"""YAML configs with ``key.sub=value`` command-line overrides.

Configs are plain nested dicts wrapped in :class:`Config` so they can be read as
attributes (``cfg.model.d_model``) and still be dumped back to YAML unchanged.
We deliberately avoid Hydra/OmegaConf: one small file is easier to reason about
and has no global state.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads ``3e-4`` as a float.

    PyYAML follows YAML 1.1, where a float needs a dot (``3.0e-4``), so plain
    ``3e-4`` would silently become the *string* "3e-4" and break learning rates.
    """


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def _yaml_load(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)  # noqa: S506 - _Loader is a SafeLoader


class Config(dict):
    """A dict whose keys are also attributes; nested dicts become Configs."""

    def __init__(self, data: Mapping[str, Any] | None = None):
        super().__init__()
        for k, v in (data or {}).items():
            self[k] = _wrap(v)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = _wrap(value)

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], Mapping):
                node[part] = Config()
            node = node[part]
        node[parts[-1]] = _wrap(value)

    def to_dict(self) -> dict[str, Any]:
        return {k: v.to_dict() if isinstance(v, Config) else copy.deepcopy(v) for k, v in self.items()}


def _wrap(v: Any) -> Any:
    if isinstance(v, Config):
        return v
    if isinstance(v, Mapping):
        return Config(v)
    if isinstance(v, list):
        return [_wrap(x) for x in v]
    return v


def _merge(base: dict, override: Mapping) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _merge(dict(out[k]), v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Config:
    """Load a YAML file. A top-level ``_base_: other.yaml`` key is merged first.

    ``overrides`` are strings like ``train.lr=3e-4``; the value is parsed as YAML
    so numbers, booleans and lists work (``model.lora.targets=[q,v]``).
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = _yaml_load(f.read()) or {}
    base = data.pop("_base_", None)
    if base is not None:
        data = _merge(load_config(path.parent / base).to_dict(), data)
    cfg = Config(data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        cfg.set_path(key.strip(), _yaml_load(raw))
    return cfg


def save_config(cfg: Mapping[str, Any], path: str | Path) -> None:
    data = cfg.to_dict() if isinstance(cfg, Config) else dict(cfg)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

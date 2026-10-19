"""Reproduce the personalisation ablation on the synthetic world (CPU, ~3 min).

Trains the base model if needed, then trains a per-user adapter three ways on a
"silent-mouthing" user and prints CER before/after for each LoRA placement.

    python scripts/ablation_personalization.py
"""

from __future__ import annotations

import json
from pathlib import Path

from punarvaani.training import train_vsr

BASE = Path("outputs/synthetic_vsr")
VARIANTS = {
    "attention+FFN only": ["lora.targets=[q_proj,v_proj,out_proj,fc1,fc2]", "lora.train_extra=[]"],
    "+ feature projections": [
        "lora.targets=[q_proj,v_proj,out_proj,fc1,fc2,proj,post_extract_proj]",
        "lora.train_extra=[]",
    ],
    "+ BatchNorm (default)": [],
}


def main() -> None:
    if not (BASE / "best.pt").exists():
        train_vsr.main(["--config", "configs/synthetic_vsr.yaml"])
    rows = {}
    for name, overrides in VARIANTS.items():
        out = f"outputs/ablation/{name.replace(' ', '_').replace('+', 'plus')}"
        res = train_vsr.main(
            ["--config", "configs/synthetic_personalize.yaml", f"train.out_dir={out}", *overrides]
        )
        rows[name] = {"cer_before": res["initial"]["cer"], "cer_after": res["best"]}
    print("\n| LoRA placement | CER before | CER after |\n|---|---|---|")
    for name, r in rows.items():
        print(f"| {name} | {r['cer_before']:.1%} | {r['cer_after']:.1%} |")
    Path("outputs/ablation/results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

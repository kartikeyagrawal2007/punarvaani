"""Train (or fine-tune) the unit HiFi-GAN vocoder.

Two uses:

1. **Multi-speaker pretraining** on the mined Hindi audio: units (from
   mHuBERT) + ECAPA speaker embedding -> waveform.
2. **Patient fine-tuning** on the cleaned old voice notes from
   :mod:`punarvaani.voice_rebuild`: same recipe, a few thousand steps, starting
   from (1). Only audio is needed - this is the step that makes Path B sound
   like *them*.

Trained on random fixed-length segments (as in HiFi-GAN), which keeps batches
uniform and the discriminators fast.

    python -m punarvaani.training.train_vocoder --config configs/synthetic_vocoder.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from punarvaani.datasets.av_dataset import read_manifest
from punarvaani.datasets.synthetic import SyntheticLipDataset, letter_tones_wav
from punarvaani.models.path_b.vocoder import Discriminators, UnitHiFiGAN, VocoderLoss
from punarvaani.preprocessing.audio import load_audio
from punarvaani.utils import get_device, get_logger, load_config, save_config, seed_everything

log = get_logger(__name__)
HOP = 320


class UnitAudioSegments(Dataset):
    """Random aligned (units, waveform) segments of ``segment_units`` units."""

    def __init__(self, items: list[dict], segment_units: int = 28, speaker_dim: int = 0):
        self.items, self.seg, self.speaker_dim = items, segment_units, speaker_dim

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        item = self.items[i]
        units, wav = item["units"], item["wav"]
        n = min(len(units), wav.numel() // HOP)
        units, wav = units[:n], wav[: n * HOP]
        if n < self.seg:  # pad short clips with the last unit / silence
            units = torch.cat([units, units[-1:].repeat(self.seg - n)])
            wav = torch.cat([wav, torch.zeros((self.seg - n) * HOP)])
            n = self.seg
        s = random.randint(0, n - self.seg)
        out = {"units": units[s : s + self.seg], "wav": wav[s * HOP : (s + self.seg) * HOP].unsqueeze(0)}
        if self.speaker_dim:
            out["speaker"] = item.get("speaker", torch.zeros(self.speaker_dim))
        return out


def synthetic_items(n: int, seed: int) -> list[dict]:
    ds = SyntheticLipDataset(n, size=8, seed=seed)
    items = []
    for i in range(n):
        _, labels = ds.render(ds.sentence(i), np.random.default_rng(i))
        items.append({"units": torch.tensor(labels).repeat_interleave(2), "wav": letter_tones_wav(labels)})
    return items


def manifest_items(path: str, speaker_dim: int) -> list[dict]:
    items = []
    for row in read_manifest(path):
        root = Path(row["_root"])
        item = {
            "units": torch.from_numpy(np.load(root / row["units"]).astype(np.int64)),
            "wav": load_audio(root / row["audio"]),
        }
        if speaker_dim and "speaker_embedding" in row:
            item["speaker"] = torch.from_numpy(np.load(root / row["speaker_embedding"])).float()
        items.append(item)
    return items


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.get_path("train.seed", 0))
    device = get_device(cfg.get("device"))
    m, t, d = cfg.model, cfg.train, cfg.data
    spk_dim = m.get("speaker_dim", 0)

    if d.kind == "synthetic":
        train_items, val_items = synthetic_items(d.get("train_samples", 128), 1), synthetic_items(8, 999)
        num_units = m.get("num_units", 32)
    else:
        train_items, val_items = (
            manifest_items(d.train_manifest, spk_dim),
            manifest_items(d.val_manifest, spk_dim),
        )
        num_units = m.get("num_units", 1000)

    if m.get("size", "full") == "tiny":
        gen, disc = UnitHiFiGAN.tiny(num_units, spk_dim, m.get("use_pitch", False)), Discriminators.tiny()
    else:
        gen = UnitHiFiGAN(num_units, speaker_dim=spk_dim, use_pitch=m.get("use_pitch", False))
        disc = Discriminators()
    if cfg.get_path("init.checkpoint"):  # fine-tuning on a patient's voice starts here
        ckpt = torch.load(cfg.init.checkpoint, map_location="cpu", weights_only=False)
        gen.load_state_dict(ckpt["generator"])
        disc.load_state_dict(ckpt["discriminator"])
    gen, disc = gen.to(device), disc.to(device)
    losses = VocoderLoss(m.get("mel_weight", 45.0)).to(device)

    opt_g = torch.optim.AdamW(gen.parameters(), t.get("lr", 2e-4), betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(disc.parameters(), t.get("lr", 2e-4), betas=(0.8, 0.99))
    gamma = t.get("lr_decay", 0.9995)
    sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma)
    loader = DataLoader(UnitAudioSegments(train_items, d.get("segment_units", 28), spk_dim),
                        batch_size=d.get("batch_size", 8), shuffle=True, drop_last=True)  # fmt: skip
    val = DataLoader(UnitAudioSegments(val_items, d.get("segment_units", 28), spk_dim), batch_size=8)

    out_dir = Path(t.get("out_dir", "outputs/vocoder"))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    history, step, max_steps = [], 0, t.get("max_steps", 1000)

    def evaluate() -> float:
        gen.eval()
        with torch.no_grad():
            vals = [
                float(losses.mel_l1(b["wav"].to(device), gen(b["units"].to(device), b.get("speaker", None))))
                for b in val
            ]
        gen.train()
        return sum(vals) / len(vals)

    history.append({"step": 0, "val_mel_l1": evaluate()})
    while step < max_steps:
        for batch in loader:
            real, units = batch["wav"].to(device), batch["units"].to(device)
            spk = batch["speaker"].to(device) if "speaker" in batch else None
            fake = gen(units, spk)
            d_loss = losses.discriminator_step(disc, real, fake)
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
            g_loss, logs = losses.generator_step(disc, real, fake)
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 10.0)
            opt_g.step()
            sched_g.step()
            sched_d.step()
            step += 1
            if step % t.get("log_every", 50) == 0:
                log.info(
                    "step=%d d=%.3f g=%.3f mel_l1=%.3f",
                    step,
                    float(d_loss),
                    float(g_loss),
                    float(logs["mel_l1"]),
                )
            if step % t.get("eval_every", 200) == 0 or step == max_steps:
                history.append({"step": step, "val_mel_l1": evaluate()})
                log.info("eval %s", history[-1])
            if step >= max_steps:
                break
    torch.save({"generator": gen.state_dict(), "discriminator": disc.state_dict(), "config": cfg.to_dict()},
               out_dir / "vocoder.pt")  # fmt: skip
    return {"history": history}


if __name__ == "__main__":
    main()

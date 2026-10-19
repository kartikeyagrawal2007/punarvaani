"""Train the hybrid's unit-ASR model: speech units -> characters (CTC). Audio-only data.

Any transcribed Hindi audio works (no video): extract units with mHuBERT-147,
pair them with the transcript. On the synthetic world, units are letter ids.

    python -m punarvaani.training.train_unit_asr --config configs/synthetic_unit_asr.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from punarvaani.eval.metrics import cer
from punarvaani.models.hybrid.guidance import UnitASR
from punarvaani.models.path_a.decoding import ctc_greedy
from punarvaani.training.train_vsr import build_data
from punarvaani.training.trainer import TrainConfig, Trainer
from punarvaani.utils import get_device, get_logger, load_config, save_config, seed_everything

log = get_logger(__name__)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    tcfg = TrainConfig.from_dict(cfg.get("train"))
    seed_everything(tcfg.seed)
    device = get_device(cfg.get("device"))
    if cfg.data.kind != "synthetic":
        cfg.data.load_units = True
    vocab, train_loader, val_loader = build_data(cfg)
    num_units = cfg.get_path("model.num_units", len(vocab))
    model = UnitASR(
        num_units, len(vocab), cfg.get_path("model.dim", 256), cfg.get_path("model.layers", 3)
    ).to(device)

    def loss_fn(m, batch):
        unit_lengths = (batch["units"] != -100).sum(1)
        loss = m.loss(
            batch["units"], unit_lengths, batch["tokens"], batch["token_lengths"], num_units, vocab.blank_id
        )
        return loss, {"ctc": loss.detach()}

    def eval_fn(m):
        refs, hyps = [], []
        for batch in val_loader:
            units = batch["units"].to(device)
            lp = m(torch.nn.functional.one_hot(units.clamp(min=0), num_units).float())
            lengths = (units != -100).sum(1) // 2
            hyps += [vocab.decode(ids) for ids in ctc_greedy(lp, lengths.cpu(), vocab.blank_id)]
            refs += batch["texts"]
        return {"cer": cer(refs, hyps)}

    Path(tcfg.out_dir).mkdir(parents=True, exist_ok=True)
    save_config(cfg, Path(tcfg.out_dir) / "config.yaml")
    trainer = Trainer(model, loss_fn, train_loader, tcfg, eval_fn, metric="cer", device=device)
    trainer.fit()
    final = trainer.evaluate()
    log.info("final %s", final)
    return {"final": final}


if __name__ == "__main__":
    main()

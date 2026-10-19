"""Train Path B's lip-to-unit model (units + mel [+ pitch] [+ aux CTC]).

The encoder can start from a Path A checkpoint (``init.encoder_from``): the
Hindi-adapted visual encoder is shared, only the heads differ.

    python -m punarvaani.training.train_units --config configs/synthetic_units.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from punarvaani.eval.metrics import edit_counts
from punarvaani.models.encoder.avhubert import EncoderConfig, VisualSpeechEncoder, load_fairseq_checkpoint
from punarvaani.models.path_b.model import LipToUnitModel
from punarvaani.models.path_b.units import dedup_units
from punarvaani.training.train_vsr import build_data
from punarvaani.training.trainer import TrainConfig, Trainer
from punarvaani.utils import (
    Config,
    count_parameters,
    get_device,
    get_logger,
    load_config,
    save_config,
    seed_everything,
)

log = get_logger(__name__)


def build_unit_model(cfg: Config, vocab_size: int) -> LipToUnitModel:
    m = cfg.model
    encoder = VisualSpeechEncoder(
        EncoderConfig.preset(m.get("encoder_preset", "tiny"), **dict(m.get("encoder_overrides") or {}))
    )
    if m.get("avhubert_checkpoint"):
        load_fairseq_checkpoint(encoder, m.avhubert_checkpoint)
    model = LipToUnitModel(
        encoder, num_units=m.get("num_units", vocab_size), hidden=m.get("hidden", 512),
        predict_mel=m.get("predict_mel", True), predict_pitch=m.get("predict_pitch", False),
        ctc_vocab_size=vocab_size if m.get("aux_ctc", True) else None, mel_weight=m.get("mel_weight", 1.0),
        ctc_weight=m.get("ctc_weight", 0.1),
    )  # fmt: skip
    src = cfg.get_path("init.encoder_from")
    if src:
        state = torch.load(src, map_location="cpu", weights_only=False)
        state = {
            k[len("encoder.") :]: v for k, v in state.get("model", state).items() if k.startswith("encoder.")
        }
        missing = model.encoder.load_state_dict(state, strict=False).missing_keys
        log.info("initialised encoder from %s (%d missing keys)", src, len(missing))
    if cfg.get_path("freeze.encoder"):
        for p in model.encoder.parameters():
            p.requires_grad = False
    return model


def unit_loss(model: LipToUnitModel, batch: dict):
    out = model(batch["video"], batch["lengths"], batch.get("units"), batch.get("mel"), batch.get("pitch"),
                batch.get("tokens"), batch.get("token_lengths"))  # fmt: skip
    return out.loss, out.logs


def make_eval_fn(loader):
    def evaluate(model: LipToUnitModel) -> dict:
        device = next(model.parameters()).device
        correct = total = 0
        errors = ref_len = 0
        for batch in loader:
            out = model(batch["video"].to(device), batch["lengths"].to(device))
            pred = out.unit_logits.argmax(-1).cpu()
            units = batch["units"]
            valid = units != -100
            correct += int(((pred == units) & valid).sum())
            total += int(valid.sum())
            for b in range(units.shape[0]):
                ref, _ = dedup_units(units[b][valid[b]])
                hyp, _ = dedup_units(pred[b][valid[b]])
                c = edit_counts(ref.tolist(), hyp.tolist())
                errors, ref_len = errors + c.errors, ref_len + c.ref_len
        return {"unit_acc": correct / max(1, total), "uer": errors / max(1, ref_len)}

    return evaluate


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
    model = build_unit_model(cfg, len(vocab)).to(device)
    log.info(
        "parameters: %d (%d trainable)", count_parameters(model), count_parameters(model, trainable_only=True)
    )
    Path(tcfg.out_dir).mkdir(parents=True, exist_ok=True)
    save_config(cfg, Path(tcfg.out_dir) / "config.yaml")
    trainer = Trainer(
        model, unit_loss, train_loader, tcfg, make_eval_fn(val_loader), metric="uer", device=device
    )
    trainer.fit()
    final = trainer.evaluate()
    log.info("final %s", final)
    return {"final": final, "history": trainer.history}


if __name__ == "__main__":
    main()

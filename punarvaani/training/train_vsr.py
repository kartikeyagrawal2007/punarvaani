"""Train the Path A recogniser: Hindi adaptation and per-user personalisation.

One script, three recipes (see ``configs/``):

1. ``synthetic_vsr.yaml`` - tiny model on the toy lip world (CPU, ~1 min).
2. ``hindi_adapt.yaml`` - AV-HuBERT-Large + ``hindi`` LoRA + new CTC/attention
   heads on the mined Hindi data. Encoder weights frozen, only adapters and
   heads train.
3. ``personalize.yaml`` - start from (2), load the ``hindi`` adapter frozen,
   add a small ``user_<id>`` adapter and train it on 10-15 min of the user's
   silent mouthing. Only the adapter is saved (a few MB per user).

Run::

    python -m punarvaani.training.train_vsr --config configs/synthetic_vsr.yaml train.max_steps=300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from punarvaani.datasets.av_dataset import AVManifestDataset, DynamicBatchSampler, VideoAugment, collate_av
from punarvaani.datasets.synthetic import SyntheticLipDataset
from punarvaani.eval.metrics import cer, wer
from punarvaani.models.encoder.avhubert import EncoderConfig, VisualSpeechEncoder, load_fairseq_checkpoint
from punarvaani.models.encoder.lora import inject_lora, load_adapter, mark_trainable, save_adapter
from punarvaani.models.path_a.decoding import ctc_greedy
from punarvaani.models.path_a.model import VSRModel
from punarvaani.text.vocab import CharVocab
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


def build_vsr(cfg: Config, vocab: CharVocab) -> VSRModel:
    m = cfg.model
    enc_cfg = EncoderConfig.preset(m.get("encoder_preset", "tiny"), **dict(m.get("encoder_overrides") or {}))
    encoder = VisualSpeechEncoder(enc_cfg)
    if m.get("avhubert_checkpoint"):
        report = load_fairseq_checkpoint(encoder, m.avhubert_checkpoint)
        log.info(
            "loaded AV-HuBERT: %d missing, %d unexpected keys",
            len(report["missing"]),
            len(report["unexpected"]),
        )
    model = VSRModel(
        encoder, len(vocab), vocab.blank_id, vocab.pad_id, vocab.bos_id, vocab.eos_id,
        decoder_layers=m.get("decoder_layers", 6), decoder_ffn=m.get("decoder_ffn"),
        ctc_weight=m.get("ctc_weight", 0.3), label_smoothing=m.get("label_smoothing", 0.1),
    )  # fmt: skip
    init = cfg.get("init") or {}
    if init.get("checkpoint"):
        state = torch.load(init["checkpoint"], map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model", state), strict=False)
    for adapter_path in init.get("adapters") or []:
        name = load_adapter(model.encoder, adapter_path)
        log.info("loaded frozen adapter %s from %s", name, adapter_path)
    return model


def configure_trainable(model: VSRModel, cfg: Config) -> int:
    lora = cfg.get("lora") or {}
    freeze = cfg.get("freeze") or {}
    if lora.get("enabled"):
        n = inject_lora(model.encoder, lora.get("name", "hindi"), lora.get("r", 16), lora.get("alpha", 32),
                        lora.get("dropout", 0.05), tuple(lora.get("targets") or ("q_proj", "v_proj", "out_proj", "fc1", "fc2")),
                        lora.get("layers"))  # fmt: skip
        log.info("LoRA %r injected into %d linears", lora.get("name", "hindi"), n)
        extra = list(lora.get("train_extra", ["ctc_head", "decoder"]))
        return mark_trainable(model, [lora.get("name", "hindi")], extra)
    for p in model.parameters():
        p.requires_grad = True
    if freeze.get("frontend"):
        for p in model.encoder.frontend_parameters():
            p.requires_grad = False
    if freeze.get("encoder"):
        for p in model.encoder.parameters():
            p.requires_grad = False
    return count_parameters(model, trainable_only=True)


def build_data(cfg: Config):
    d = cfg.data
    if d.kind == "synthetic":
        common = dict(
            size=d.get("size", 44),
            noise=d.get("noise", 0.05),
            confusable=[tuple(x) for x in d.get("confusable", [])],
            **dict(d.get("style") or {}),
        )
        train = SyntheticLipDataset(d.get("train_samples", 512), seed=1, **common)
        val = SyntheticLipDataset(d.get("val_samples", 64), seed=999, **common)
        vocab = train.vocab
        train_loader = DataLoader(train, batch_size=d.get("batch_size", 16), shuffle=True,
                                  collate_fn=lambda b: collate_av(b, vocab.pad_id), num_workers=0)  # fmt: skip
        val_loader = DataLoader(val, batch_size=32, collate_fn=lambda b: collate_av(b, vocab.pad_id))
        return vocab, train_loader, val_loader
    vocab = CharVocab()
    modes = d.get("modes")
    units = d.get("load_units", False)
    train = AVManifestDataset(
        d.train_manifest, vocab, VideoAugment(train=True), units, d.get("max_frames", 500), modes
    )
    val = AVManifestDataset(d.val_manifest, vocab, VideoAugment(train=False), units, modes=modes)
    sampler = DynamicBatchSampler(
        [train.num_frames(i) for i in range(len(train))], d.get("max_frames_per_batch", 2000)
    )
    train_loader = DataLoader(train, batch_sampler=sampler, collate_fn=lambda b: collate_av(b, vocab.pad_id),
                              num_workers=d.get("num_workers", 4))  # fmt: skip
    val_loader = DataLoader(
        val, batch_size=8, collate_fn=lambda b: collate_av(b, vocab.pad_id), num_workers=2
    )
    return vocab, train_loader, val_loader


def vsr_loss(model: VSRModel, batch: dict):
    out = model(batch["video"], batch["lengths"], batch["tokens"], batch["token_lengths"])
    return out.loss, {"loss": out.loss.detach(), "ctc": out.ctc_loss.detach(), "att": out.att_loss.detach()}


def make_eval_fn(loader, vocab: CharVocab, max_batches: int | None = None):
    def evaluate(model: VSRModel) -> dict:
        refs, hyps = [], []
        device = next(model.parameters()).device
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            out = model(batch["video"].to(device), batch["lengths"].to(device))
            for ids in ctc_greedy(out.ctc_log_probs, batch["lengths"], vocab.blank_id):
                hyps.append(vocab.decode(ids))
            refs.extend(batch["texts"])
        return {"cer": cer(refs, hyps), "wer": wer(refs, hyps), "example": f"{refs[0]} -> {hyps[0]}"}

    return evaluate


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume")
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    tcfg = TrainConfig.from_dict(cfg.get("train"))
    seed_everything(tcfg.seed)
    device = get_device(cfg.get("device"))

    vocab, train_loader, val_loader = build_data(cfg)
    model = build_vsr(cfg, vocab)
    n_train = configure_trainable(model, cfg)
    log.info("parameters: %d total, %d trainable (%.2f%%)", count_parameters(model), n_train,
             100 * n_train / count_parameters(model))  # fmt: skip
    model.to(device)
    out_dir = Path(tcfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    vocab.save(out_dir / "vocab.json")

    trainer = Trainer(model, vsr_loss, train_loader, tcfg, make_eval_fn(val_loader, vocab, cfg.get_path("eval.max_batches")),
                      metric="cer", device=device)  # fmt: skip
    if args.resume:
        trainer.resume(args.resume)
    initial = trainer.evaluate() if cfg.get_path("eval.at_start", False) else {}
    if initial:
        log.info("before training %s", initial)
    trainer.fit()
    lora = cfg.get("lora") or {}
    if lora.get("enabled"):
        save_adapter(model.encoder, lora.get("name", "hindi"), out_dir / f"adapter_{lora.get('name', 'hindi')}.pt",
                     meta={"config": cfg.to_dict()})  # fmt: skip
    final = trainer.evaluate()
    log.info("final %s", final)
    return {"initial": initial, "final": final, "best": trainer.best, "history": trainer.history}


if __name__ == "__main__":
    main()

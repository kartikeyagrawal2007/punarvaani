import pytest
import torch
from torch.utils.data import DataLoader

from punarvaani.datasets.av_dataset import DynamicBatchSampler, collate_av
from punarvaani.datasets.synthetic import SyntheticLipDataset
from punarvaani.training import train_vsr
from punarvaani.training.trainer import TrainConfig, Trainer, freeze_frozen_batchnorm, warmup_cosine
from punarvaani.utils import Config


def test_warmup_cosine_shape():
    assert warmup_cosine(0, 10, 100, 0.1) == pytest.approx(0.1)
    assert warmup_cosine(9, 10, 100, 0.1) == pytest.approx(1.0)
    assert warmup_cosine(100, 10, 100, 0.1) == pytest.approx(0.1)


def test_dynamic_batches_respect_frame_budget_and_cover_everything():
    lengths = [5, 50, 7, 49, 8, 51, 6, 30] * 10
    sampler = DynamicBatchSampler(lengths, max_frames=120)
    batches = list(sampler)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    assert all(max(lengths[i] for i in b) * len(b) <= 120 or len(b) == 1 for b in batches)


def test_synthetic_collation_shapes():
    ds = SyntheticLipDataset(4, size=32)
    batch = collate_av([ds[i] for i in range(4)], ds.vocab.pad_id)
    t = batch["video"].shape[1]
    assert batch["video"].shape[2:] == (32, 32)
    assert batch["units"].shape == (4, 2 * t) and batch["mel"].shape == (4, 4 * t, 80)
    assert (batch["tokens"] != ds.vocab.pad_id).sum(1).tolist() == batch["token_lengths"].tolist()


def test_frozen_batchnorm_stays_in_eval():
    bn = torch.nn.BatchNorm2d(3)
    for p in bn.parameters():
        p.requires_grad = False
    model = torch.nn.Sequential(bn).train()
    freeze_frozen_batchnorm(model)
    assert not bn.training


def test_trainer_runs_saves_and_resumes(tmp_path):
    ds = SyntheticLipDataset(8, size=32)
    loader = DataLoader(ds, batch_size=4, collate_fn=lambda b: collate_av(b, ds.vocab.pad_id))
    cfg = Config({"model": {"encoder_preset": "tiny", "decoder_layers": 1, "decoder_ffn": 32}})
    model = train_vsr.build_vsr(cfg, ds.vocab)
    tcfg = TrainConfig(max_steps=3, warmup_steps=1, log_every=1, eval_every=3, out_dir=str(tmp_path))
    trainer = Trainer(model, train_vsr.vsr_loss, loader, tcfg, train_vsr.make_eval_fn(loader, ds.vocab))
    history = trainer.fit()
    assert any("eval" in h for h in history) and (tmp_path / "last.pt").exists()
    again = Trainer(model, train_vsr.vsr_loss, loader, tcfg)
    again.resume(tmp_path / "last.pt")
    assert again.step == 3


@pytest.mark.slow
def test_tiny_vsr_learns_the_synthetic_task(tmp_path):
    result = train_vsr.main([
        "--config", "configs/synthetic_vsr.yaml", "train.max_steps=150", "train.eval_every=150",
        "data.train_samples=256", f"train.out_dir={tmp_path}",
    ])  # fmt: skip
    assert result["final"]["cer"] < 0.35


def test_unit_and_vocoder_training_smoke(tmp_path):
    from punarvaani.training import train_units, train_vocoder

    res = train_units.main([
        "--config", "configs/synthetic_units.yaml", "train.max_steps=2", "train.eval_every=2",
        "data.train_samples=8", "data.val_samples=4", f"train.out_dir={tmp_path / 'u'}",
    ])  # fmt: skip
    assert res["final"]["uer"] >= 0 and 0 <= res["final"]["unit_acc"] <= 1  # UER can exceed 1 (insertions)
    res = train_vocoder.main([
        "--config", "configs/synthetic_vocoder.yaml", "train.max_steps=2", "train.eval_every=2",
        "data.train_samples=8", f"train.out_dir={tmp_path / 'v'}",
    ])  # fmt: skip
    assert (tmp_path / "v" / "vocoder.pt").exists() and len(res["history"]) == 2


def test_unit_asr_training_smoke(tmp_path):
    from punarvaani.training import train_unit_asr

    res = train_unit_asr.main([
        "--config", "configs/synthetic_unit_asr.yaml", "train.max_steps=2", "train.eval_every=2",
        "data.train_samples=8", "data.val_samples=4", f"train.out_dir={tmp_path}",
    ])  # fmt: skip
    assert 0 <= res["final"]["cer"]

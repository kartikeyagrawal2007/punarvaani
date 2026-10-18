import torch

from punarvaani.utils import lengths_to_padding_mask, load_config


def test_load_config_with_base_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("model:\n  d_model: 64\n  layers: 2\ntrain:\n  lr: 0.001\n")
    (tmp_path / "child.yaml").write_text("_base_: base.yaml\nmodel:\n  layers: 4\n")
    cfg = load_config(tmp_path / "child.yaml", ["train.lr=3e-4", "model.lora.targets=[q, v]"])
    assert cfg.model.d_model == 64
    assert cfg.model.layers == 4
    assert abs(cfg.train.lr - 3e-4) < 1e-12
    assert cfg.model.lora.targets == ["q", "v"]
    assert cfg.get_path("model.missing", 7) == 7


def test_padding_mask():
    mask = lengths_to_padding_mask(torch.tensor([3, 1]))
    assert mask.tolist() == [[False, False, False], [False, True, True]]

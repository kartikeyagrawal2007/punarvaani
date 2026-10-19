import json

from punarvaani import cli
from punarvaani.eval import evaluate
from punarvaani.training import train_vsr


def test_train_then_evaluate_with_phrases_and_lm(tmp_path):
    run = tmp_path / "run"
    train_vsr.main(["--config", "configs/synthetic_vsr.yaml", "train.max_steps=3", "train.eval_every=3",
                    "data.train_samples=8", "data.val_samples=8", f"train.out_dir={run}"])  # fmt: skip
    phrases = tmp_path / "phrases.txt"
    phrases.write_text("कमल\nनल दल\n", encoding="utf-8")
    out = tmp_path / "eval.json"
    res = evaluate.main(["--run", str(run), "--phrases", str(phrases), "--lm-corpus", str(phrases), "--beam", "3",
                         "--out", str(out)])  # fmt: skip
    assert {"greedy", "beam", "latency_ms"} <= set(res) and res["n"] == 8
    saved = json.loads(out.read_text(encoding="utf-8"))
    cmp = evaluate.compare(saved, saved)
    assert cmp["rate_a"] == cmp["rate_b"]


def test_personalised_run_reloads_its_adapter(tmp_path):
    base = tmp_path / "base"
    train_vsr.main(["--config", "configs/synthetic_vsr.yaml", "train.max_steps=2", "train.eval_every=2",
                    "data.train_samples=8", "data.val_samples=4", f"train.out_dir={base}"])  # fmt: skip
    pers = tmp_path / "pers"
    train_vsr.main(["--config", "configs/synthetic_personalize.yaml", "train.max_steps=2", "train.eval_every=2",
                    "data.train_samples=8", "data.val_samples=4", f"init.checkpoint={base / 'best.pt'}",
                    f"train.out_dir={pers}"])  # fmt: skip
    _, _, model = evaluate.load_run(pers)
    assert any("user_demo" in n for n, _ in model.named_parameters())
    assert (pers / "adapter_user_demo.pt").exists()


def test_cli_dispatches_training(tmp_path):
    cli.main(["train-vsr", "--config", "configs/synthetic_vsr.yaml", "train.max_steps=1", "train.eval_every=1",
              "data.train_samples=4", "data.val_samples=4", f"train.out_dir={tmp_path}"])  # fmt: skip
    assert (tmp_path / "last.pt").exists()

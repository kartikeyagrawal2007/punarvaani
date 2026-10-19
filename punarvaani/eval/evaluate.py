"""Evaluate a trained Path A run: open-vocabulary decoding, phrase mode, latency.

    python -m punarvaani.eval.evaluate --run outputs/hindi_adapt --manifest data/test.jsonl \\
        --phrases configs/phrases_hi.txt --out outputs/hindi_adapt/eval.json

Reports CER / WER / viseme error rate for greedy and beam(+char LM) decoding,
phrase-mode top-1 accuracy and rejection rate (on clips whose text is in the
phrase bank), and per-stage latency percentiles. Hypotheses are saved so two
systems can be compared with :func:`compare` (paired bootstrap).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from punarvaani.datasets.av_dataset import AVManifestDataset, VideoAugment, collate_av
from punarvaani.datasets.synthetic import SyntheticLipDataset
from punarvaani.eval.metrics import LatencyMeter, cer, paired_bootstrap, viseme_error_rate, wer
from punarvaani.models.path_a.decoding import CharNGramLM, ctc_greedy, ctc_prefix_beam_search
from punarvaani.models.path_a.phrase_mode import PhraseBank
from punarvaani.text.hindi import normalize
from punarvaani.text.vocab import CharVocab
from punarvaani.training.train_vsr import build_vsr
from punarvaani.utils import Config, load_config


def load_run(run_dir: str | Path, checkpoint: str = "best.pt"):
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    vocab = CharVocab.load(run_dir / "vocab.json")
    # Weights come from the checkpoint below, but frozen adapters (e.g. 'hindi' under a
    # personalised run) must be re-created so their slots exist in the state dict.
    cfg.init = Config({"adapters": list(cfg.get_path("init.adapters") or [])})
    model = build_vsr(cfg, vocab)
    lora = cfg.get("lora") or {}
    if lora.get("enabled"):
        from punarvaani.training.train_vsr import configure_trainable

        configure_trainable(model, cfg)  # recreate adapter slots so the state dict matches
    state = torch.load(run_dir / checkpoint, map_location="cpu", weights_only=False)
    result = model.load_state_dict(state["model"], strict=False)
    bad = [k for k in (*result.missing_keys, *result.unexpected_keys) if "lora" in k]
    if bad:  # silently dropping adapter weights would evaluate the wrong model
        raise RuntimeError(f"adapter weights do not match the run config: {bad[:5]}")
    return cfg, vocab, model.eval()


def evaluate(model, vocab: CharVocab, loader, phrase_bank: PhraseBank | None = None, lm: CharNGramLM | None = None,
             beam: int = 8, lm_weight: float = 0.3) -> dict:  # fmt: skip
    refs, greedy, beamed = [], [], []
    phrase_hits = phrase_total = phrase_rejected = 0
    meter = LatencyMeter()
    for batch in loader:
        with meter.measure("encoder_batch"):
            out = model(batch["video"], batch["lengths"])
        greedy += [
            vocab.decode(ids) for ids in ctc_greedy(out.ctc_log_probs, batch["lengths"], vocab.blank_id)
        ]
        for b in range(len(batch["texts"])):
            lp = out.ctc_log_probs[b, : int(batch["lengths"][b])]
            t0 = time.perf_counter()
            hyps = ctc_prefix_beam_search(lp, beam, vocab.blank_id, lm, lm_weight if lm else 0.0)
            meter.add("beam_search", (time.perf_counter() - t0) * 1e3)
            beamed.append(vocab.decode(hyps[0].tokens) if hyps else "")
            ref = normalize(batch["texts"][b])
            if phrase_bank is not None and ref in phrase_bank.phrases:
                t0 = time.perf_counter()
                m = phrase_bank.match(lp, use_prior=False)
                meter.add("phrase_match", (time.perf_counter() - t0) * 1e3)
                phrase_total += 1
                phrase_rejected += not m.accepted
                phrase_hits += m.accepted and m.phrase == ref
        refs += batch["texts"]
    result = {
        "n": len(refs),
        "greedy": {
            "cer": cer(refs, greedy),
            "wer": wer(refs, greedy),
            "ver": viseme_error_rate(refs, greedy),
        },
        "beam": {"cer": cer(refs, beamed), "wer": wer(refs, beamed), "ver": viseme_error_rate(refs, beamed)},
        "latency_ms": meter.summary(),
        "hypotheses": [{"ref": r, "greedy": g, "beam": h} for r, g, h in zip(refs, greedy, beamed)],
    }
    if phrase_total:
        result["phrase_mode"] = {"n": phrase_total, "top1_accuracy": phrase_hits / phrase_total,
                                 "rejection_rate": phrase_rejected / phrase_total,
                                 "accuracy_when_accepted": phrase_hits / max(1, phrase_total - phrase_rejected)}  # fmt: skip
    return result


def compare(eval_a: dict, eval_b: dict, key: str = "beam", unit: str = "word") -> dict:
    """Paired bootstrap between two evaluations on the same test set."""
    refs = [h["ref"] for h in eval_a["hypotheses"]]
    assert refs == [h["ref"] for h in eval_b["hypotheses"]], "different test sets"
    return paired_bootstrap(
        refs, [h[key] for h in eval_a["hypotheses"]], [h[key] for h in eval_b["hypotheses"]], unit
    )


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--manifest", help="test manifest; omit to use the synthetic validation set")
    ap.add_argument("--phrases", help="phrase bank (.txt or .json)")
    ap.add_argument("--lm-corpus", help="text file to train a character LM for beam search")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    cfg, vocab, model = load_run(args.run)
    if args.manifest:
        ds = AVManifestDataset(args.manifest, vocab, VideoAugment(train=False))
    else:
        d = cfg.data
        ds = SyntheticLipDataset(d.get("val_samples", 64), size=d.get("size", 44), noise=d.get("noise", 0.05), seed=999,
                                 **dict(d.get("style") or {}))  # fmt: skip
    loader = DataLoader(ds, batch_size=8, collate_fn=lambda b: collate_av(b, vocab.pad_id))
    bank = PhraseBank.load(args.phrases, vocab) if args.phrases else None
    lm = None
    if args.lm_corpus:
        lines = [
            ln
            for ln in Path(args.lm_corpus).read_text(encoding="utf-8").splitlines()
            if ln and not ln.startswith("#")
        ]
        lm = CharNGramLM(5, len(vocab)).fit(vocab.encode(ln) for ln in lines)
    with torch.no_grad():
        result = evaluate(model, vocab, loader, bank, lm, args.beam)
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = {k: v for k, v in result.items() if k != "hypotheses"}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return result


if __name__ == "__main__":
    main()

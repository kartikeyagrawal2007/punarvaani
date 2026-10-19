"""Evaluation metrics: WER/CER/VER, significance testing, MOS, speaker verification, latency.

* **WER / CER** - word / character error rate = (substitutions + deletions +
  insertions) / reference length, summed over the whole test set (corpus
  level, not an average of per-utterance rates, which over-weights short
  clips). Text is normalised at the ``loose`` level first so spelling
  conventions (nukta, chandrabindu) don't count as reading errors.
* **VER** - viseme error rate: CER computed on viseme sequences. If CER is high
  but VER is low, the model sees the right mouth shapes and the remaining
  errors are look-alikes that context (LM / LLM) should fix. That split tells
  us where to spend effort.
* **Paired bootstrap** (Koehn 2004) - is system A really better than B on
  this test set, or is it noise? Resample utterances with replacement, recompute
  both WERs on each resample, count how often A wins. Essential for ablations on
  small Hindi test sets where 1-2% WER differences are within noise.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from punarvaani.text.hindi import normalize
from punarvaani.text.phonemes import text_to_visemes


@dataclass
class EditCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    ref_len: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        return self.errors / max(1, self.ref_len)

    def __add__(self, other: EditCounts) -> EditCounts:
        return EditCounts(self.substitutions + other.substitutions, self.deletions + other.deletions,
                          self.insertions + other.insertions, self.ref_len + other.ref_len)  # fmt: skip


def edit_counts(ref: Sequence, hyp: Sequence) -> EditCounts:
    """Levenshtein alignment with S/D/I breakdown (ties prefer substitution)."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]), d[i - 1, j] + 1, d[i, j - 1] + 1)
    s = de = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            de, i = de + 1, i - 1
        else:
            ins, j = ins + 1, j - 1
    return EditCounts(int(s), de, ins, n)


def _units(text: str, unit: str) -> list:
    text = normalize(text, level="loose")
    if unit == "word":
        return text.split()
    if unit == "char":
        return list(text.replace(" ", ""))
    if unit == "viseme":
        return text_to_visemes(text, word_sep=None)
    raise ValueError(unit)


def error_rate(refs: Sequence[str], hyps: Sequence[str], unit: str = "word") -> EditCounts:
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps differ in length")
    total = EditCounts()
    for r, h in zip(refs, hyps):
        total = total + edit_counts(_units(r, unit), _units(h, unit))
    return total


def wer(refs, hyps) -> float:
    return error_rate(refs, hyps, "word").rate


def cer(refs, hyps) -> float:
    return error_rate(refs, hyps, "char").rate


def viseme_error_rate(refs, hyps) -> float:
    return error_rate(refs, hyps, "viseme").rate


def paired_bootstrap(refs: Sequence[str], hyps_a: Sequence[str], hyps_b: Sequence[str], unit: str = "word",
                     n_resamples: int = 1000, seed: int = 0) -> dict:  # fmt: skip
    """Returns WERs, the 95% CI of (A - B) and the fraction of resamples where A is *not* better (a p-value)."""
    ua = [edit_counts(_units(r, unit), _units(h, unit)) for r, h in zip(refs, hyps_a)]
    ub = [edit_counts(_units(r, unit), _units(h, unit)) for r, h in zip(refs, hyps_b)]
    rng = random.Random(seed)
    n = len(refs)
    diffs = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        ref_len = sum(ua[i].ref_len for i in idx) or 1
        diffs.append((sum(ua[i].errors for i in idx) - sum(ub[i].errors for i in idx)) / ref_len)
    diffs.sort()
    total_a, total_b = sum(ua, EditCounts()), sum(ub, EditCounts())
    return {
        "rate_a": total_a.rate,
        "rate_b": total_b.rate,
        "diff_ci95": (diffs[int(0.025 * n_resamples)], diffs[int(0.975 * n_resamples) - 1]),
        "p_a_not_better": sum(d >= 0 for d in diffs) / n_resamples,
    }


def mos_summary(ratings: Sequence[float], n_resamples: int = 2000, seed: int = 0) -> dict:
    """Mean opinion score with a bootstrap 95% CI (ratings 1-5)."""
    r = np.asarray(ratings, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = rng.choice(r, size=(n_resamples, len(r)), replace=True).mean(1)
    return {
        "mos": float(r.mean()),
        "ci95": (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))),
        "n": len(r),
    }


def cosine_similarity(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def equal_error_rate(scores: Sequence[float], labels: Sequence[int]) -> tuple[float, float]:
    """EER and its threshold for verification trials (label 1 = same speaker)."""
    s, y = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    order = np.argsort(-s)
    s, y = s[order], y[order]
    pos, neg = max(1, y.sum()), max(1, (1 - y).sum())
    tpr = np.cumsum(y) / pos
    fpr = np.cumsum(1 - y) / neg
    fnr = 1 - tpr
    k = int(np.argmin(np.abs(fnr - fpr)))
    return float((fnr[k] + fpr[k]) / 2), float(s[k])


class LatencyMeter:
    """Collect per-stage wall-clock times; report p50/p90/p99 in ms."""

    def __init__(self):
        self.samples: dict[str, list[float]] = {}

    @contextmanager
    def measure(self, stage: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.samples.setdefault(stage, []).append((time.perf_counter() - t) * 1e3)

    def add(self, stage: str, ms: float) -> None:
        self.samples.setdefault(stage, []).append(ms)

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for stage, xs in self.samples.items():
            a = np.asarray(xs)
            out[stage] = {"p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
                          "p99": float(np.percentile(a, 99)), "n": len(a)}  # fmt: skip
        return out


def real_time_factor(processing_s: float, audio_s: float) -> float:
    return processing_s / audio_s if audio_s > 0 else math.inf

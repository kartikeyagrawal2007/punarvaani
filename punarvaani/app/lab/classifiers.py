"""Two few-shot phrase readers trained from a handful of the user's own takes.

* :class:`DTWClassifier` - no training at all. Dynamic time warping lines up a
  new take with every recorded take (stretching time so a slow "पानी" matches a
  fast one) and picks the phrase with the closest match. Works from 1 take per
  phrase, which makes it the baseline every learned model has to beat.
* :class:`NeuralClassifier` - a small neural network (a temporal CNN over
  lip-shape features) trained in under a minute on a laptop CPU. Data
  augmentation (time stretch, trimming, scale and noise jitter) stretches the
  few takes into thousands of slightly different ones.

Both return a :class:`Prediction` with a confidence, and both can *reject*:
phrase mode should stay silent rather than say the wrong thing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from punarvaani.app.lab.features import with_deltas


@dataclass
class Prediction:
    label: str | None  # None when rejected
    confidence: float
    top: list[tuple[str, float]]
    rejected: bool
    model: str


class Normalizer:
    """Per-dimension z-score fitted on all training frames (shape and motion end up on one scale)."""

    def fit(self, seqs: Sequence[np.ndarray]) -> Normalizer:
        frames = np.concatenate(seqs)
        self.mean, self.std = frames.mean(0), frames.std(0) + 1e-4
        return self

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        return ((seq - self.mean) / self.std).astype(np.float32)


def dtw_distances(query: np.ndarray, templates: Sequence[np.ndarray]) -> np.ndarray:
    """Length-normalised DTW distance from ``query`` (n, F) to each template (m_b, F).

    All templates are aligned at once: the cost tensor is (B, n, M) and the
    dynamic programme sweeps anti-diagonals (cells with equal i + j depend only
    on the previous two diagonals), so there are n + M vectorised steps instead
    of n x M Python ones per template.
    """
    n, lengths = len(query), np.array([len(t) for t in templates])
    big_m = int(lengths.max())
    tpl = np.zeros((len(templates), big_m, query.shape[1]), dtype=np.float64)
    for b, t in enumerate(templates):
        tpl[b, : len(t)] = t
    q = query.astype(np.float64)
    sq = (q**2).sum(-1)[None, :, None] + (tpl**2).sum(-1)[:, None, :] - 2 * np.einsum("nf,bmf->bnm", q, tpl)
    cost = np.sqrt(np.maximum(sq, 0.0))
    acc = np.full((len(templates), n + 1, big_m + 1), np.inf)
    acc[:, 0, 0] = 0.0
    for k in range(2, n + big_m + 1):
        i = np.arange(max(1, k - big_m), min(n, k - 1) + 1)
        j = k - i
        best = np.minimum(np.minimum(acc[:, i - 1, j - 1], acc[:, i - 1, j]), acc[:, i, j - 1])
        acc[:, i, j] = cost[:, i - 1, j - 1] + best
    return acc[np.arange(len(templates)), n, lengths] / (n + lengths)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _decide(classes: list[str], probs: np.ndarray, reject_below: float, model: str, force_reject=False):
    order = np.argsort(-probs)
    top = [(classes[i], float(probs[i])) for i in order[:5]]
    conf = float(probs[order[0]])
    rejected = force_reject or conf < reject_below
    return Prediction(None if rejected else classes[order[0]], conf, top, rejected, model)


class DTWClassifier:
    name = "dtw"

    def __init__(self, reject_below: float = 0.5):
        self.reject_below = reject_below

    def fit(self, seqs: Sequence[np.ndarray], labels: Sequence[str]) -> DTWClassifier:
        feats = [with_deltas(s) for s in seqs]
        self.norm = Normalizer().fit(feats)
        self.templates = [self.norm(f) for f in feats]
        self.labels = list(labels)
        self.classes = sorted(set(labels))
        # Typical distance between two takes of the *same* phrase sets the confidence scale.
        within = []
        for k, t in enumerate(self.templates):
            same = [x for x, lab in zip(self.templates, self.labels) if lab == self.labels[k] and x is not t]
            if same:
                within.append(dtw_distances(t, same).min())
        self.tau = float(np.median(within)) if within else 1.0
        return self

    def class_distances(self, seq: np.ndarray) -> np.ndarray:
        d = dtw_distances(self.norm(with_deltas(seq)), self.templates)
        return np.array([d[[lab == c for lab in self.labels]].min() for c in self.classes])

    def predict(self, seq: np.ndarray) -> Prediction:
        d = self.class_distances(seq)
        probs = _softmax(-d / (0.15 * self.tau))
        # Far from *every* recorded phrase: something else was said, so stay silent.
        unknown = d.min() > 2.5 * self.tau
        return _decide(self.classes, probs, self.reject_below, self.name, force_reject=unknown)


class _ConvNet(nn.Module):
    """Temporal CNN: dilated 1-D convolutions over time (receptive field ~1 s), then masked mean+max pooling.

    Same job as an RNN over the frames, but every time step is computed in
    parallel, so it trains several times faster on a laptop CPU.
    """

    def __init__(self, in_dim: int, num_classes: int, hidden: int = 96, dropout: float = 0.2):
        super().__init__()
        layers, ch = [], in_dim
        for d in (1, 2, 4, 1):
            layers += [nn.Conv1d(ch, hidden, 5, padding=2 * d, dilation=d), nn.BatchNorm1d(hidden), nn.GELU(),
                       nn.Dropout(dropout)]  # fmt: skip
            ch = hidden
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(2 * hidden, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.body(x.transpose(1, 2)).transpose(1, 2)  # (B, T, hidden)
        mask = (torch.arange(x.shape[1])[None, :] < lengths[:, None]).unsqueeze(-1)
        mean = (h * mask).sum(1) / lengths[:, None]
        mx = h.masked_fill(~mask, -1e4).max(1).values
        return self.out(torch.cat([mean, mx], -1))


def augment(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One random variant of a (T, 80) take: new speed, trimmed edges, slightly different mouth size."""
    t = len(seq)
    new_t = max(4, int(round(t * rng.uniform(0.8, 1.25))))
    src = np.linspace(0, t - 1, new_t)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (src - lo)[:, None]
    out = seq[lo] * (1 - w) + seq[hi] * w
    cut = int(0.1 * new_t)
    a, b = rng.integers(0, cut + 1), rng.integers(0, cut + 1)
    out = out[a : new_t - b] if new_t - a - b >= 4 else out
    out = out * rng.uniform(0.9, 1.1) + rng.normal(0, 0.02, out.shape[1])
    return (out + rng.normal(0, 0.01, out.shape)).astype(np.float32)


def _pad(seqs: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(s) for s in seqs])
    x = torch.zeros(len(seqs), int(lengths.max()), seqs[0].shape[1])
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.from_numpy(s)
    return x, lengths


@dataclass
class NeuralClassifier:
    reject_below: float = 0.5
    steps: int = 300
    batch: int = 16
    lr: float = 3e-3
    seed: int = 0
    progress: Callable[[int, float], None] | None = None  # called as (step, loss) every 10 steps
    losses: list[float] = field(default_factory=list)
    name: str = "neural"

    def fit(self, seqs: Sequence[np.ndarray], labels: Sequence[str]) -> NeuralClassifier:
        progress = self.progress
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        self.classes = sorted(set(labels))
        y_all = np.array([self.classes.index(lab) for lab in labels])
        self.norm = Normalizer().fit([with_deltas(s) for s in seqs])
        self.net = _ConvNet(seqs[0].shape[1] * 2, len(self.classes))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr, total_steps=self.steps, pct_start=0.15
        )
        self.net.train()
        self.losses = []
        for step in range(self.steps):
            idx = rng.integers(0, len(seqs), self.batch)
            x, lengths = _pad([self.norm(with_deltas(augment(seqs[i], rng))) for i in idx])
            loss = F.cross_entropy(self.net(x, lengths), torch.from_numpy(y_all[idx]), label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step()
            sched.step()
            self.losses.append(loss.item())
            if progress and (step % 10 == 0 or step == self.steps - 1):
                progress(step + 1, self.losses[-1])
        self.net.eval()
        return self

    @torch.no_grad()
    def predict(self, seq: np.ndarray) -> Prediction:
        x, lengths = _pad([self.norm(with_deltas(seq))])
        probs = torch.softmax(self.net(x, lengths), -1)[0].numpy()
        return _decide(self.classes, probs, self.reject_below, self.name)


@dataclass
class CVReport:
    """Cross-validated accuracy: every take is predicted by a model that never saw it."""

    accuracy: float  # correct / all takes (a rejection counts as not correct)
    answered_accuracy: float | None  # correct / takes it did not reject
    coverage: float  # share of takes it answered
    n: int
    confusion: dict[str, dict[str, int]]  # true phrase -> predicted phrase (or "(not sure)") -> count

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def cross_validate(make: Callable[[], object], seqs: Sequence[np.ndarray], labels: Sequence[str],
                   folds: int | None = None) -> CVReport | None:  # fmt: skip
    """``folds=None`` is leave-one-out. Takes are assigned to folds round-robin within each phrase."""
    counts = Counter(labels)
    if len(counts) < 2 or min(counts.values()) < 2:
        return None  # need two phrases with two takes each before held-out numbers mean anything
    seen: Counter = Counter()
    fold_of = []
    for lab in labels:
        fold_of.append(seen[lab])
        seen[lab] += 1
    k = folds or max(counts.values())
    fold_of = [f % k for f in fold_of]
    confusion: dict[str, dict[str, int]] = {}
    correct = answered = 0
    for f in range(k):
        train = [i for i, g in enumerate(fold_of) if g != f]
        test = [i for i, g in enumerate(fold_of) if g == f]
        if not test or len({labels[i] for i in train}) < 2:
            continue
        clf = make().fit([seqs[i] for i in train], [labels[i] for i in train])
        for i in test:
            p = clf.predict(seqs[i])
            got = p.label or "(not sure)"
            confusion.setdefault(labels[i], {}).setdefault(got, 0)
            confusion[labels[i]][got] += 1
            answered += not p.rejected
            correct += p.label == labels[i]
    n = sum(sum(v.values()) for v in confusion.values())
    if n == 0:
        return None
    return CVReport(correct / n, correct / answered if answered else None, answered / n, n, confusion)

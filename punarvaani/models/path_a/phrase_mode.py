"""Phrase mode: closed-set recognition over the user's 200-500 everyday phrases.

Open-vocabulary lip reading is ~30% WER even in English. But a patient's daily
needs are a small, known set ("मुझे पानी चाहिए", "दर्द हो रहा है", ...). Picking
one of N known phrases is a far easier problem than writing any sentence, so
this is the guaranteed-to-work fallback.

How we score: the CTC head gives per-frame character posteriors. For every
phrase ``y`` we compute the exact sequence likelihood

    log P(y | video) = log Σ_{alignments a -> y} Π_t P(a_t | video)

which is just ``-ctc_loss`` (the CTC forward algorithm). All phrases are scored
in one batched call. Advantages over training a classifier head:

* adding/editing phrases needs **no retraining** (a family member can add
  "बेटी को बुलाओ" and it works immediately);
* it reuses the same model as open-vocabulary mode;
* the scores are real likelihoods, so we can combine them with a **prior**
  (how often the user says each phrase, context) by Bayes' rule and do
  **rejection** when nothing fits (instead of confidently saying the wrong
  thing, which for a patient is worse than saying nothing).

Rejection uses two tests: the phrase posterior must be confident, and the best
phrase must explain the video nearly as well as unconstrained decoding does (a
likelihood-ratio test: if the free decoder finds something far more likely,
the person said something outside the set).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from punarvaani.models.path_a.decoding import best_path_score
from punarvaani.text.hindi import normalize
from punarvaani.text.phonemes import text_to_visemes
from punarvaani.text.vocab import CharVocab


@dataclass
class PhraseMatch:
    phrase: str
    index: int
    posterior: float
    log_likelihood: float
    accepted: bool
    reason: str = ""
    top_k: list[tuple[str, float]] = field(default_factory=list)


class PhraseBank:
    def __init__(self, phrases: Sequence[str], vocab: CharVocab, counts: Sequence[float] | None = None,
                 prior_strength: float = 1.0):  # fmt: skip
        self.vocab = vocab
        self.phrases = [normalize(p) for p in phrases]
        if len(set(self.phrases)) != len(self.phrases):
            raise ValueError("duplicate phrases after normalisation")
        self.targets = [vocab.encode(p, already_normalized=True) for p in self.phrases]
        # Dirichlet-smoothed usage prior: every phrase starts with pseudo-count
        # ``prior_strength`` and gains 1 each time the user confirms it.
        self.counts = list(counts) if counts is not None else [0.0] * len(self.phrases)
        self.prior_strength = prior_strength
        self.visemes = [text_to_visemes(p) for p in self.phrases]

    def __len__(self) -> int:
        return len(self.phrases)

    def log_prior(self) -> torch.Tensor:
        c = torch.tensor(self.counts, dtype=torch.float64) + self.prior_strength
        return (c / c.sum()).log()

    def confirm(self, index: int) -> None:
        """Called when the user (or a carer) confirms a phrase was right."""
        self.counts[index] += 1.0

    def log_likelihoods(self, log_probs: torch.Tensor, length: int | None = None) -> torch.Tensor:
        """``log_probs``: (T, V) CTC log-posteriors for one utterance -> (N,) log P(phrase | video)."""
        lp = log_probs[:length] if length is not None else log_probs
        t, n = lp.shape[0], len(self.phrases)
        target_lengths = torch.tensor([len(x) for x in self.targets])
        targets = torch.cat([torch.tensor(x, dtype=torch.long) for x in self.targets])
        losses = F.ctc_loss(
            lp.float().unsqueeze(1).expand(t, n, -1), targets, torch.full((n,), t, dtype=torch.long),
            target_lengths, blank=self.vocab.blank_id, reduction="none", zero_infinity=False,
        )  # fmt: skip
        return -losses.double()  # inf loss (phrase too long for the clip) -> -inf

    def match(
        self,
        log_probs: torch.Tensor,
        length: int | None = None,
        use_prior: bool = True,
        temperature: float = 1.0,
        min_posterior: float = 0.5,
        max_ratio_per_frame: float = 0.5,
        top_k: int = 5,
    ) -> PhraseMatch:
        ll = self.log_likelihoods(log_probs, length)
        scores = ll / temperature + (self.log_prior() if use_prior else 0.0)
        post = torch.softmax(scores, 0)
        best = int(post.argmax())
        frames = length if length is not None else log_probs.shape[0]
        # Likelihood ratio vs. unconstrained decoding, normalised per frame. The best
        # single path is a lower bound on the free-decoding likelihood.
        free = best_path_score(log_probs, length)
        ratio = (free - float(ll[best])) / max(1, frames)
        order = post.argsort(descending=True)[:top_k].tolist()
        top = [(self.phrases[i], float(post[i])) for i in order]
        reason = ""
        if not math.isfinite(float(ll[best])):
            reason = "no phrase fits the clip length"
        elif float(post[best]) < min_posterior:
            reason = f"ambiguous: top posterior {float(post[best]):.2f} < {min_posterior}"
        elif ratio > max_ratio_per_frame:
            reason = f"out of set: free decoding is {ratio:.2f} nats/frame more likely"
        return PhraseMatch(
            self.phrases[best], best, float(post[best]), float(ll[best]), not reason, reason, top
        )

    def match_text(self, hypothesis: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Fallback for text-only recognisers: rank phrases by viseme edit distance.

        Distances are on visemes, so "बानी" (a misread of पानी) still matches
        "पानी" perfectly - the two look identical on the lips.
        """
        hyp = text_to_visemes(normalize(hypothesis))
        scored = [
            (p, 1.0 - _edit_distance(hyp, v) / max(len(hyp), len(v), 1))
            for p, v in zip(self.phrases, self.visemes)
        ]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

    def save(self, path: str | Path) -> None:
        data = {"phrases": self.phrases, "counts": self.counts, "prior_strength": self.prior_strength}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, vocab: CharVocab) -> PhraseBank:
        path = Path(path)
        if path.suffix == ".txt":
            lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
            return cls([ln for ln in lines if ln and not ln.startswith("#")], vocab)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["phrases"], vocab, data.get("counts"), data.get("prior_strength", 1.0))


def _edit_distance(a: Sequence, b: Sequence) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

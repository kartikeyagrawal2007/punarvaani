"""Decoding for Path A: CTC greedy, CTC prefix beam search with LM fusion, attention rescoring.

CTC prefix beam search (Graves 2012; Hannun et al. 2014). A CTC output is a
frame-level path like ``"--कक-म--ल"`` (``-`` = blank) that collapses to "कमल".
Many paths collapse to the same text, so the probability of a text is a *sum*
over paths. Greedy decoding picks the single best path and ignores that sum;
prefix beam search keeps, for every candidate prefix, two numbers:

    p_b   probability of all paths giving this prefix and ending in blank
    p_nb  ... ending in the prefix's last character

The split matters because a repeated character needs a blank in between:
from "क" ending in blank, emitting क gives "कक"; from "क" ending in क, emitting
क again stays "क". Everything is kept in log space for numerical stability.

Shallow fusion adds ``λ · log P_LM(char | prefix)`` whenever a character is
appended, plus a length bonus to counter the LM's preference for short outputs.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

NEG_INF = -float("inf")


def _logsumexp(*xs: float) -> float:
    m = max(xs)
    if m == NEG_INF:
        return NEG_INF
    return m + math.log(sum(math.exp(x - m) for x in xs))


def ctc_greedy(
    log_probs: torch.Tensor, lengths: torch.Tensor | None = None, blank: int = 0
) -> list[list[int]]:
    """(B, T, V) -> best-path token ids with repeats merged and blanks removed."""
    best = log_probs.argmax(-1)
    out = []
    for b in range(best.shape[0]):
        t = int(lengths[b]) if lengths is not None else best.shape[1]
        seq, prev = [], None
        for tok in best[b, :t].tolist():
            if tok != prev and tok != blank:
                seq.append(tok)
            prev = tok
        out.append(seq)
    return out


def best_path_score(log_probs: torch.Tensor, length: int | None = None) -> float:
    """Log probability of the single best CTC path for one utterance (T, V)."""
    lp = log_probs[:length] if length is not None else log_probs
    return float(lp.max(-1).values.sum())


class CharNGramLM:
    """Interpolated Witten-Bell character n-gram LM.

    Small, dependency-free and fast enough to call inside beam search. Trained
    on Hindi text (transcripts of the mined dataset, Wikipedia, the user's own
    phrase list), it teaches the decoder which character sequences are Hindi.
    """

    def __init__(self, order: int = 5, vocab_size: int | None = None):
        self.order = order
        self.counts: list[dict[tuple, dict[int, int]]] = [
            defaultdict(lambda: defaultdict(int)) for _ in range(order)
        ]
        self.vocab: set[int] = set()
        self.vocab_size = vocab_size
        self.bos = -1

    def fit(self, sequences: Iterable[Sequence[int]]) -> CharNGramLM:
        for seq in sequences:
            padded = [self.bos] * (self.order - 1) + list(seq)
            for i in range(self.order - 1, len(padded)):
                tok = padded[i]
                self.vocab.add(tok)
                for n in range(self.order):  # n = context length
                    ctx = tuple(padded[i - n : i]) if n else ()
                    self.counts[n][ctx][tok] += 1
        self.vocab_size = max(self.vocab_size or 0, len(self.vocab))
        self._cache: dict = {}
        return self

    def prob(self, context: Sequence[int], token: int) -> float:
        context = (
            ([self.bos] * (self.order - 1) + list(context))[-(self.order - 1) :] if self.order > 1 else []
        )
        key = (tuple(context), token)
        if key in self._cache:
            return self._cache[key]
        p = 1.0 / max(1, self.vocab_size)  # uniform floor
        for n in range(self.order):
            ctx = tuple(context[len(context) - n :]) if n else ()
            followers = self.counts[n].get(ctx)
            if not followers:
                continue
            total, types = sum(followers.values()), len(followers)
            lam = total / (total + types)  # Witten-Bell: trust contexts seen often with few continuations
            p = lam * followers.get(token, 0) / total + (1 - lam) * p
        self._cache[key] = p
        return p

    def log_prob(self, context: Sequence[int], token: int) -> float:
        return math.log(self.prob(context, token))

    def score(self, seq: Sequence[int]) -> float:
        return sum(self.log_prob(seq[:i], tok) for i, tok in enumerate(seq))


@dataclass
class Hypothesis:
    tokens: list[int]
    score: float  # combined score used for ranking
    ctc_score: float  # log P_ctc(tokens | video), summed over alignments
    lm_score: float = 0.0
    att_score: float = 0.0


def ctc_prefix_beam_search(
    log_probs: torch.Tensor,
    beam: int = 10,
    blank: int = 0,
    lm: CharNGramLM | None = None,
    lm_weight: float = 0.0,
    length_bonus: float = 0.0,
    prune_top_k: int = 20,
    prune_threshold: float = -12.0,
) -> list[Hypothesis]:
    """n-best list for one utterance; ``log_probs`` is (T, V) log-softmax output.

    Per frame we only expand the ``prune_top_k`` most likely tokens and skip
    tokens with log-prob below ``prune_threshold``; that keeps the search
    O(T · beam · k) instead of O(T · beam · V).
    """
    lp = log_probs.detach().float().cpu()
    beams: dict[tuple, tuple[float, float, float]] = {(): (0.0, NEG_INF, 0.0)}  # prefix -> (p_b, p_nb, lm)

    for t in range(lp.shape[0]):
        frame = lp[t]
        top_v, top_i = frame.topk(min(prune_top_k, frame.numel()))
        cand = [(int(i), float(v)) for v, i in zip(top_v, top_i) if v > prune_threshold or int(i) == blank]
        nxt: dict[tuple, list[float]] = defaultdict(lambda: [NEG_INF, NEG_INF, 0.0])
        for prefix, (p_b, p_nb, lm_s) in beams.items():
            total = _logsumexp(p_b, p_nb)
            for tok, p in cand:
                if tok == blank:
                    e = nxt[prefix]
                    e[0] = _logsumexp(e[0], total + p)
                    e[2] = lm_s
                    continue
                last = prefix[-1] if prefix else None
                new = prefix + (tok,)
                e = nxt[new]
                lm_add = lm_weight * lm.log_prob(prefix, tok) if lm is not None and lm_weight else 0.0
                if tok == last:
                    # repeat: only paths ending in blank can start a new copy of the char
                    e[1] = _logsumexp(e[1], p_b + p)
                    same = nxt[prefix]
                    same[1] = _logsumexp(same[1], p_nb + p)
                    same[2] = lm_s
                else:
                    e[1] = _logsumexp(e[1], total + p)
                e[2] = lm_s + lm_add
        ranked = sorted(
            nxt.items(),
            key=lambda kv: _logsumexp(kv[1][0], kv[1][1]) + kv[1][2] + length_bonus * len(kv[0]),
            reverse=True,
        )
        beams = {k: tuple(v) for k, v in ranked[:beam]}

    hyps = []
    for prefix, (p_b, p_nb, lm_s) in beams.items():
        ctc = _logsumexp(p_b, p_nb)
        hyps.append(Hypothesis(list(prefix), ctc + lm_s + length_bonus * len(prefix), ctc, lm_s))
    return sorted(hyps, key=lambda h: h.score, reverse=True)


def attention_rescore(model, enc: torch.Tensor, enc_mask: torch.Tensor | None, hyps: list[Hypothesis],
                      ctc_weight: float = 0.3) -> list[Hypothesis]:  # fmt: skip
    """Re-rank an n-best list with ``ctc_weight·CTC + (1-ctc_weight)·attention (+ LM part)``."""
    if not hyps:
        return hyps
    att = model.decoder_score(enc, enc_mask, [h.tokens for h in hyps])
    for h, a in zip(hyps, att.tolist()):
        h.att_score = a
        h.score = ctc_weight * h.ctc_score + (1 - ctc_weight) * a + h.lm_score
    return sorted(hyps, key=lambda h: h.score, reverse=True)

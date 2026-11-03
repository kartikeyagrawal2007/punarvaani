"""
Error metrics shared by training, evaluation and tests.

All rates are corpus-level: total edits divided by total reference length.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Sequence


def levenshtein_distance(seq1: Sequence, seq2: Sequence) -> int:
    """Plain Levenshtein edit distance between two sequences."""
    m, n = len(seq1), len(seq2)
    if m == 0:
        return n
    if n == 0:
        return m

    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        a = seq1[i - 1]
        for j in range(1, n + 1):
            cost = 0 if a == seq2[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def normalize_devanagari(text: str) -> str:
    """Keep only Devanagari characters and whitespace."""
    if not text:
        return ""
    return re.sub(r"[^\u0900-\u097F\s]", "", text).strip()


def corpus_cer_wer(predictions: Iterable[str], references: Iterable[str]) -> Dict[str, float]:
    """
    Compute corpus-level CER and WER over paired prediction/reference strings.

    Returns keys: ``cer``, ``wer`` (percentages), ``char_edits``, ``ref_chars``,
    ``word_edits``, ``ref_words``.
    """
    total_char_edits = 0
    total_ref_chars = 0
    total_word_edits = 0
    total_ref_words = 0
    n = 0

    for pred, ref in zip(predictions, references):
        n += 1
        pred_clean = normalize_devanagari(pred)
        ref_clean = normalize_devanagari(ref)
        total_char_edits += levenshtein_distance(list(pred_clean), list(ref_clean))
        total_ref_chars += len(list(ref_clean))
        total_word_edits += levenshtein_distance(pred_clean.split(), ref_clean.split())
        total_ref_words += len(ref_clean.split())

    cer = (total_char_edits / total_ref_chars * 100.0) if total_ref_chars else 0.0
    wer = (total_word_edits / total_ref_words * 100.0) if total_ref_words else 0.0
    return {
        "cer": cer,
        "wer": wer,
        "char_edits": float(total_char_edits),
        "ref_chars": float(total_ref_chars),
        "word_edits": float(total_word_edits),
        "ref_words": float(total_ref_words),
        "num_samples": float(n),
    }

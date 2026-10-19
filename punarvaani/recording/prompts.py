"""Prompt selection and session planning for volunteer / patient recordings.

Two kinds of prompts per person:

* **phrases** - their actual phrase bank (what phrase mode must recognise);
* **balanced sentences** - chosen from a Hindi text corpus to cover as many
  phoneme pairs (diphones) as possible, so 10-15 minutes of recording teaches
  the per-user adapter every mouth transition, not just the 50 most common.
  Greedy set cover: repeatedly take the sentence that adds the most *new*
  diphones per word (shorter sentences are cheaper to record).

Each prompt is recorded **spoken aloud, then silently mouthed**: the paired
recordings let us measure the silent-vs-spoken gap (research question 2) and
give spoken clips whose audio can supervise the silent ones (same sentence,
same person, time-aligned by DTW on lip landmarks).
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from punarvaani.text.hindi import normalize
from punarvaani.text.phonemes import text_to_phonemes


def diphones(sentence: str) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for word in text_to_phonemes(normalize(sentence)):
        seq = ["#", *word, "#"]
        out.update(zip(seq, seq[1:]))
    return out


def select_balanced_sentences(corpus: Iterable[str], n: int, max_words: int = 12) -> list[str]:
    candidates = []
    for s in corpus:
        s = normalize(s)
        if s and len(s.split()) <= max_words:
            candidates.append((s, diphones(s)))
    covered: set[tuple[str, str]] = set()
    chosen: list[str] = []
    while candidates and len(chosen) < n:
        best = max(
            range(len(candidates)),
            key=lambda i: len(candidates[i][1] - covered) / len(candidates[i][0].split()),
        )
        sentence, units = candidates.pop(best)
        if not units - covered:
            break
        chosen.append(sentence)
        covered |= units
    return chosen


def coverage(sentences: Sequence[str], reference: Iterable[str]) -> float:
    """Fraction of the reference corpus' diphones covered by ``sentences``."""
    ref: set = set().union(*(diphones(s) for s in reference))
    have: set = set().union(*(diphones(s) for s in sentences)) if sentences else set()
    return len(ref & have) / max(1, len(ref))


@dataclass
class Prompt:
    index: int
    text: str
    mode: str  # "spoken" or "silent"
    kind: str  # "phrase" or "sentence"


def plan_session(
    phrases: Sequence[str], sentences: Sequence[str], repeats: int = 1, seed: int = 0
) -> list[Prompt]:
    """Shuffled prompt list; each item appears spoken then silent, back to back."""
    items = [(p, "phrase") for p in phrases for _ in range(repeats)] + [(s, "sentence") for s in sentences]
    random.Random(seed).shuffle(items)
    prompts = []
    for text, kind in items:
        for mode in ("spoken", "silent"):
            prompts.append(Prompt(len(prompts), normalize(text), mode, kind))
    return prompts


def save_plan(prompts: Sequence[Prompt], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps([asdict(p) for p in prompts], ensure_ascii=False, indent=1), encoding="utf-8"
    )


def load_phrases(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]

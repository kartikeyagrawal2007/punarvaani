"""Cut long transcribed videos into training clips at natural pauses.

Input: word-level timestamps from the ASR (IndicWhisper / IndicConformer).
Output: clips of 1-12 s that start and end in pauses, so no word is cut in
half (a half word at a clip edge is a guaranteed label error for CTC).

Greedy rule: keep adding words while the clip stays under ``max_s``; prefer to
close the clip at the longest pause once it is past ``target_s``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class Word:
    text: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class Clip:
    start: float
    end: float
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def min_confidence(self) -> float:
        return min((w.confidence for w in self.words), default=0.0)


def segment_words(words: Sequence[Word], min_s: float = 1.0, max_s: float = 12.0, target_s: float = 6.0,
                  break_pause_s: float = 0.35, pad_s: float = 0.12) -> list[Clip]:  # fmt: skip
    """Group words into clips; ``pad_s`` of context is added around each clip (lips move before sound starts)."""
    clips: list[Clip] = []
    current: list[Word] = []

    def close():
        if current:
            start, end = current[0].start - pad_s, current[-1].end + pad_s
            if end - start >= min_s:
                clips.append(Clip(max(0.0, start), end, list(current)))
            current.clear()

    for i, w in enumerate(words):
        if current and (w.end - current[0].start) > max_s:
            close()
        current.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        pause = (nxt.start - w.end) if nxt else float("inf")
        length = w.end - current[0].start
        if pause >= 1.5 * break_pause_s or (length >= target_s and pause >= break_pause_s):
            close()
    close()
    return clips

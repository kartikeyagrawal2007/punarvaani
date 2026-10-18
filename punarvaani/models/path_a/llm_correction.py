"""Context correction of lip-reading hypotheses.

The visual model gets the *mouth shapes* right far more often than the exact
words: पानी (paanii) and बानी (baanii) are the same lip movie. Context resolves
that: a patient in a hospital bed saying "मुझे ___ चाहिए" almost certainly wants
पानी. Two correctors share one interface:

* :class:`ClaudeCorrector` - an LLM sees the n-best list with CTC scores, the
  user's situation, recent conversation and phrase list, and an explicit
  briefing on which Devanagari letters are visually interchangeable. It must
  answer with JSON (structured output), and it is told to stay close to the
  visual evidence: replace letters only *within* a viseme group, never invent
  content.
* :class:`LexiconCorrector` - offline and deterministic: snaps each word to the
  closest known word *at the viseme level* and lets a word-bigram model pick
  among look-alikes. Works with no network (hospital Wi-Fi!) and is the
  fallback whenever the LLM call fails, times out or is refused.

Latency note: an LLM round trip is several hundred ms to seconds, so it runs
only on open-vocabulary utterances; phrase mode never waits for it.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from punarvaani.text.hindi import normalize
from punarvaani.text.phonemes import CONFUSABLE_LETTER_GROUPS, phonemes_to_visemes, word_to_phonemes
from punarvaani.utils import get_logger

log = get_logger(__name__)


@dataclass
class CorrectionContext:
    setting: str = "घर"  # e.g. "अस्पताल का वार्ड", "घर"
    user_profile: str = ""  # free text: name, family members, condition, habits
    recent_turns: list[str] = field(default_factory=list)  # what was said around this utterance
    phrases: list[str] = field(default_factory=list)  # the user's phrase bank
    vocabulary: list[str] = field(default_factory=list)  # names/words the user often uses


@dataclass
class Correction:
    text: str
    confidence: float
    source: str  # "llm", "lexicon", "passthrough"
    chosen_index: int = -1  # which n-best entry it was based on
    note: str = ""


class Corrector(Protocol):
    def correct(self, nbest: Sequence[tuple[str, float]], context: CorrectionContext) -> Correction: ...


def _edit_distance(a: Sequence, b: Sequence) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def word_visemes(word: str) -> tuple[str, ...]:
    return tuple(phonemes_to_visemes(word_to_phonemes(word)))


class LexiconCorrector:
    """Viseme-level nearest-word snapping plus a word-bigram LM (Viterbi over candidates)."""

    def __init__(self, corpus: Iterable[str], max_viseme_distance: float = 0.34, lm_weight: float = 1.0,
                 candidates_per_word: int = 8):  # fmt: skip
        self.unigrams: Counter[str] = Counter()
        self.bigrams: Counter[tuple[str, str]] = Counter()
        for sentence in corpus:
            words = ["<s>", *normalize(sentence).split(), "</s>"]
            self.unigrams.update(words[1:-1])
            self.bigrams.update(zip(words, words[1:]))
        self.by_visemes: dict[tuple[str, ...], list[str]] = {}
        for w in self.unigrams:
            self.by_visemes.setdefault(word_visemes(w), []).append(w)
        self.total = sum(self.unigrams.values())
        self.sentence_count = sum(c for (p, _), c in self.bigrams.items() if p == "<s>")
        self.max_viseme_distance = max_viseme_distance
        self.lm_weight = lm_weight
        self.k = candidates_per_word

    def _log_bigram(self, prev: str, word: str) -> float:
        # Bigram interpolated with a smoothed unigram (Witten-Bell-like weight): crude but adequate.
        uni = (self.unigrams.get(word, 0) + 0.1) / (self.total + 0.1 * (len(self.unigrams) + 1))
        prev_count = self.unigrams.get(prev, 0) if prev != "<s>" else self.sentence_count
        lam = prev_count / (prev_count + 5.0)
        big = self.bigrams.get((prev, word), 0) / prev_count if prev_count else 0.0
        return math.log(lam * big + (1 - lam) * uni)

    def candidates(self, word: str) -> list[tuple[str, float]]:
        """Known words that look like ``word``, with a visual cost in [0, 1]."""
        vis = word_visemes(word)
        scored = []
        for key, words in self.by_visemes.items():
            d = _edit_distance(vis, key) / max(len(vis), len(key), 1)
            if d <= self.max_viseme_distance:
                scored.extend((w, d) for w in words)
        scored.sort(key=lambda x: (x[1], -self.unigrams[x[0]]))
        if word in self.unigrams and all(w != word for w, _ in scored[: self.k]):
            scored.insert(0, (word, 0.0))
        return scored[: self.k] or [(word, 0.0)]

    def correct_sentence(self, text: str) -> tuple[str, float]:
        words = normalize(text).split()
        if not words:
            return "", 0.0
        # Viterbi: state = candidate word; score = -visual_cost * 10 + lm_weight * log P(w | prev)
        beams = {"<s>": (0.0, [])}
        for w in words:
            new: dict[str, tuple[float, list[str]]] = {}
            for cand, cost in self.candidates(w):
                best = max(
                    (
                        (s - 10.0 * cost + self.lm_weight * self._log_bigram(p, cand), path)
                        for p, (s, path) in beams.items()
                    ),
                    key=lambda x: x[0],
                )
                new[cand] = (best[0], [*best[1], cand])
            beams = new
        score, path = max(((s + self.lm_weight * self._log_bigram(p, "</s>"), path) for p, (s, path) in beams.items()),
                          key=lambda x: x[0])  # fmt: skip
        return " ".join(path), score

    def correct(self, nbest: Sequence[tuple[str, float]], context: CorrectionContext) -> Correction:
        if not nbest:
            return Correction("", 0.0, "passthrough")
        best_i, best_text, best_score = 0, "", -math.inf
        for i, (hyp, visual_score) in enumerate(nbest[:5]):
            text, lm_score = self.correct_sentence(hyp)
            total = visual_score + lm_score
            if total > best_score:
                best_i, best_text, best_score = i, text, total
        changed = best_text != normalize(nbest[best_i][0])
        return Correction(
            best_text, 0.5 if changed else 0.7, "lexicon", best_i, "snapped to lexicon" if changed else ""
        )


SYSTEM_PROMPT = """You help a person who cannot produce sound speak again. A camera reads their lips \
and a lip-reading model turns the lip movements into Hindi text. Lip reading is ambiguous: letters \
in the same group below look IDENTICAL on the lips (voicing, aspiration and nasality are invisible), \
so the model often confuses them:

{groups}

Vowel length (इ/ई, उ/ऊ) and anusvara/chandrabindu are also hard to see.

You receive the model's top hypotheses with scores (higher = more likely visually). Write the \
sentence the person most plausibly said, in Hindi (Devanagari), using:
1. the visual evidence: keep the sequence of lip shapes; you may swap letters only within a group, \
fix vowel length, and fix spacing. Do not add or remove ideas, and do not answer or continue the sentence;
2. the person's situation, recent conversation and usual phrases.
If one of their usual phrases fits the lip shapes, prefer it. If nothing sensible fits, return the \
top hypothesis unchanged with low confidence.

About this person:
Setting: {setting}
Profile: {profile}
Words and names they often use: {vocabulary}
Their usual phrases:
{phrases}"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "the corrected Hindi sentence"},
        "chosen_index": {"type": "integer", "description": "index of the hypothesis it was based on"},
        "confidence": {"type": "number", "description": "0 to 1"},
    },
    "required": ["text", "chosen_index", "confidence"],
    "additionalProperties": False,
}


def build_messages(nbest: Sequence[tuple[str, float]], context: CorrectionContext) -> tuple[str, str]:
    """Return (system, user) prompts. The system prompt only depends on the user, so it caches well."""
    system = SYSTEM_PROMPT.format(
        groups="\n".join(f"- {g}" for g in CONFUSABLE_LETTER_GROUPS),
        setting=context.setting or "-",
        profile=context.user_profile or "-",
        vocabulary=", ".join(context.vocabulary) or "-",
        phrases="\n".join(f"- {p}" for p in context.phrases) or "-",
    )
    lines = [f"{i}. {text}  (score {score:.1f})" for i, (text, score) in enumerate(nbest)]
    recent = "\n".join(f"- {t}" for t in context.recent_turns[-6:]) or "-"
    user = f"Recent conversation:\n{recent}\n\nLip-reading hypotheses:\n" + "\n".join(lines)
    return system, user


class ClaudeCorrector:
    """LLM correction through the Claude API (``pip install anthropic``).

    Credentials come from the environment (``ANTHROPIC_API_KEY`` or an
    ``ant auth login`` profile). Any failure falls back to ``fallback`` so the
    device keeps talking even without network.
    """

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low", timeout_s: float = 8.0,
                 fallback: Corrector | None = None, client=None):  # fmt: skip
        self.model, self.effort, self.timeout_s = model, effort, timeout_s
        self.fallback = fallback
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(timeout=self.timeout_s, max_retries=1)
        return self._client

    def _fallback(self, nbest, context, note: str) -> Correction:
        log.warning("LLM correction unavailable (%s); using fallback", note)
        if self.fallback is not None:
            out = self.fallback.correct(nbest, context)
            out.note = f"{note}; {out.note}".strip("; ")
            return out
        return Correction(normalize(nbest[0][0]) if nbest else "", 0.3, "passthrough", 0, note)

    def correct(self, nbest: Sequence[tuple[str, float]], context: CorrectionContext) -> Correction:
        if not nbest:
            return Correction("", 0.0, "passthrough")
        system, user = build_messages(nbest, context)
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=1024,
                # Lip-to-speech is latency-bound: low effort keeps the round trip short.
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
                },
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                # If a safety classifier declines, let the API retry on a suitable model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except Exception as e:  # network, auth, rate limit, timeout: never block the patient
            return self._fallback(nbest, context, f"{type(e).__name__}: {e}")
        if getattr(response, "stop_reason", None) == "refusal":
            return self._fallback(nbest, context, "refused")
        text = next((b.text for b in response.content if getattr(b, "type", "") == "text"), "")
        try:
            data = json.loads(text)
            out_text = normalize(str(data["text"]))
        except (json.JSONDecodeError, KeyError, TypeError):
            return self._fallback(nbest, context, "unparseable response")
        if not out_text:
            return self._fallback(nbest, context, "empty response")
        idx = int(data.get("chosen_index", 0))
        conf = float(min(1.0, max(0.0, data.get("confidence", 0.5))))
        return Correction(out_text, conf, "llm", idx if 0 <= idx < len(nbest) else 0)

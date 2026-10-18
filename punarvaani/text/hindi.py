"""Hindi (Devanagari) text normalisation.

Lip-reading targets must be *what was said*, written one consistent way. Raw
transcripts (IndicWhisper output, news subtitles) disagree on several things
that do not change the spoken words, and every disagreement adds noise to the
CTC targets and inflates CER. :func:`normalize` removes those differences.

Two levels:

* ``strict`` (default) keeps every spoken distinction and is used for training
  targets and for TTS input.
* ``loose`` additionally merges distinctions that many speakers and writers do
  not make consistently and that are invisible on the lips anyway: nukta
  (ज़ vs ज), chandrabindu vs anusvara (ँ vs ं). Use it when scoring CER so the
  metric measures reading errors, not spelling conventions.
"""

from __future__ import annotations

import re
import unicodedata

# Unicode code points we keep. Everything else (Latin, punctuation, emoji) is
# dropped or mapped. The order here also defines the vocabulary order, so do
# not reorder: appending is fine, reordering breaks old checkpoints.
INDEPENDENT_VOWELS = "अआइईउऊऋएऐओऔऍऑ"
CONSONANTS = "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"
VOWEL_SIGNS = "ािीुूृेैोौॅॉ"  # matras
SIGNS = "ंँः़्"  # anusvara, chandrabindu, visarga, virama (halant), nukta
DEVANAGARI_LETTERS = INDEPENDENT_VOWELS + CONSONANTS + VOWEL_SIGNS + SIGNS

NUKTA = "़"
VIRAMA = "्"
ANUSVARA = "ं"
CHANDRABINDU = "ँ"

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
# Zero-width joiner/non-joiner, soft hyphen, BOM: change rendering, not speech.
_INVISIBLE = dict.fromkeys(map(ord, "‌‍­﻿"), None)
_KEEP = set(DEVANAGARI_LETTERS) | {" "}

# 1..99 in Hindi are irregular and must be listed (इक्कीस is not "बीस एक").
_ONES_TO_99 = (
    "शून्य एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस "
    "बीस इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस तैंतीस चौंतीस "
    "पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस "
    "अड़तालीस उनचास पचास इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ साठ इकसठ बासठ तिरसठ "
    "चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर "
    "अठहत्तर उन्यासी अस्सी इक्यासी बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी नब्बे इक्यानवे "
    "बानवे तिरानवे चौरानवे पचानवे छियानवे सत्तानवे अट्ठानवे निन्यानवे"
).split()
assert len(_ONES_TO_99) == 100

# Indian numbering: crore (10^7), lakh (10^5), thousand, hundred.
_SCALES = ((10**7, "करोड़"), (10**5, "लाख"), (1000, "हज़ार"), (100, "सौ"))


def number_to_words(n: int) -> str:
    """Spell a non-negative integer in Hindi using the Indian numbering system.

    >>> number_to_words(2024)
    'दो हज़ार चौबीस'
    >>> number_to_words(150000)
    'एक लाख पचास हज़ार'
    """
    if n < 0:
        return "माइनस " + number_to_words(-n)
    if n < 100:
        return _ONES_TO_99[n]
    words: list[str] = []
    for value, name in _SCALES:
        if n >= value:
            q, n = divmod(n, value)
            words += [number_to_words(q), name]
    if n:
        words.append(_ONES_TO_99[n])
    return " ".join(words)


def _spell_numbers(text: str) -> str:
    text = text.translate(_DEVANAGARI_DIGITS)
    # Drop thousands separators first so "1,50,000" is one number.
    text = re.sub(r"(?<=\d)[,](?=\d)", "", text)
    return re.sub(r"\d+", lambda m: f" {number_to_words(int(m.group()))} ", text)


def normalize(text: str, level: str = "strict") -> str:
    """Normalise Hindi text for training targets (``strict``) or scoring (``loose``)."""
    if level not in ("strict", "loose"):
        raise ValueError(f"level must be 'strict' or 'loose', got {level!r}")
    # NFC also *decomposes* the precomposed nukta letters (क़ U+0958 ...) into
    # base + U+093C, because they are composition exclusions. So after this step
    # "क़" is always two code points, whichever way the source typed it.
    text = unicodedata.normalize("NFC", text).translate(_INVISIBLE)
    text = _spell_numbers(text)
    text = "".join(ch if ch in _KEEP else " " for ch in text)
    if level == "loose":
        text = text.replace(NUKTA, "").replace(CHANDRABINDU, ANUSVARA)
    return " ".join(text.split())


def is_mostly_devanagari(text: str, threshold: float = 0.9) -> bool:
    """True if at least ``threshold`` of the letters are Devanagari.

    The data engine uses this to drop code-mixed clips (lots of English words),
    because the vocabulary is Devanagari only.
    """
    letters = [
        ch for ch in unicodedata.normalize("NFC", text) if ch.isalpha() or unicodedata.category(ch) == "Mn"
    ]
    if not letters:
        return False
    dev = sum(1 for ch in letters if "ऀ" <= ch <= "ॿ")
    return dev / len(letters) >= threshold

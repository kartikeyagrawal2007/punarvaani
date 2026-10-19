"""Rule-based Hindi grapheme-to-phoneme conversion and phoneme-to-viseme mapping.

Why this exists: lip reading is fundamentally ambiguous. A *viseme* is a group of
phonemes that look the same on the lips. In Hindi, प/फ/ब/भ/म (p, ph, b, bh, m)
are one viseme: voicing, aspiration and nasality happen in the throat and nose,
which the camera cannot see. So पल (pal), बल (bal) and मल (mal) are the same lip
movie. The visual model cannot separate them; context can. This module lets
the rest of the system *reason* about that ambiguity:

* phrase mode and the lexicon corrector match words at the viseme level,
* the LLM prompt is told which letters are visually interchangeable,
* evaluation reports a viseme error rate next to CER, which tells us whether an
  error was "the model saw the wrong mouth shape" or "the model picked the wrong
  word among look-alikes".

Hindi spelling is close to phonemic, with one big exception: **schwa deletion**.
Every consonant letter carries an inherent vowel "a" (ə), but it is often not
pronounced: कमल is kamal (not kamala), कमला is kamlā (not kamalā). We use the
standard rule (Ohala 1983, as used in most Hindi TTS front-ends):

1. delete the word-final inherent schwa;
2. scanning right to left, delete a medial schwa in the context  V C _ C V
   (it sits between two single consonants that both border vowels).

This is right for most native words; loanwords and compounds have exceptions,
which is acceptable for viseme purposes (a deleted vs kept schwa changes one
short "A" viseme, not the consonant shapes).
"""

from __future__ import annotations

import unicodedata

from punarvaani.text.hindi import ANUSVARA, CHANDRABINDU, NUKTA, VIRAMA

SCHWA = "a"

_CONSONANT_PHONES = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "c", "छ": "ch", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "T", "ठ": "Th", "ड": "D", "ढ": "Dh", "ण": "N",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "sh", "ष": "Sh", "स": "s", "ह": "h",
}  # fmt: skip
# Consonant + nukta: Perso-Arabic and flapped sounds.
_NUKTA_PHONES = {"क": "q", "ख": "x", "ग": "G", "ज": "z", "ड": "R", "ढ": "Rh", "फ": "f", "य": "y"}
_VOWEL_PHONES = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऍ": "ae", "ऑ": "aw",
}  # fmt: skip
_MATRA_PHONES = {
    "ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॅ": "ae", "ॉ": "aw",
}  # fmt: skip
VOWELS = set(_VOWEL_PHONES.values()) | {SCHWA}

# Place of articulation, used to turn anusvara into the matching nasal (अंक -> ank -> "ng k").
_PLACE = {
    **dict.fromkeys(["k", "kh", "g", "gh", "q", "x", "G"], "ng"),
    **dict.fromkeys(["c", "ch", "j", "jh", "z"], "ny"),
    **dict.fromkeys(["T", "Th", "D", "Dh"], "N"),
    **dict.fromkeys(["t", "th", "d", "dh", "s", "l"], "n"),
    **dict.fromkeys(["p", "ph", "b", "bh", "m"], "m"),
}

# Viseme classes. Voicing and aspiration are invisible, and most tongue positions
# are hidden behind the teeth, so the lips mainly show: closed (P), lower lip to
# teeth (F), rounded/protruded (U, CH), spread (I), open (A), and "tongue
# somewhere inside" (T for the front, K for the back of the mouth).
VISEME_OF = {
    **dict.fromkeys(["p", "ph", "b", "bh", "m"], "P"),
    **dict.fromkeys(["f", "v"], "F"),
    **dict.fromkeys(
        ["t", "th", "d", "dh", "n", "l", "s", "z", "r", "T", "Th", "D", "Dh", "N", "R", "Rh"], "T"
    ),
    **dict.fromkeys(["c", "ch", "j", "jh", "ny", "sh", "Sh", "y"], "CH"),
    **dict.fromkeys(["k", "kh", "g", "gh", "ng", "q", "x", "G", "h"], "K"),
    **dict.fromkeys(["a", "aa", "ae", "aw"], "A"),
    **dict.fromkeys(["i", "ii", "e", "ai"], "I"),
    **dict.fromkeys(["u", "uu", "o", "au"], "U"),
    "~": "",  # nasalisation is not visible
}
VISEME_INVENTORY = ("P", "F", "T", "CH", "K", "A", "I", "U")


def _word_to_units(word: str) -> list[tuple[str, str]]:
    """Split a Devanagari word into ("C"/"V"/"N", phone) units before schwa deletion.

    Every consonant letter emits ("C", phone) followed by its vowel: a matra,
    nothing (if a virama follows) or the inherent schwa ("V", "a", marked "S").
    """
    units: list[tuple[str, str]] = []
    chars = list(unicodedata.normalize("NFC", word))
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch in _CONSONANT_PHONES:
            phone = _CONSONANT_PHONES[ch]
            if i + 1 < len(chars) and chars[i + 1] == NUKTA:
                phone = _NUKTA_PHONES.get(ch, phone)
                i += 1
            units.append(("C", phone))
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            if nxt == VIRAMA:
                i += 1
            elif nxt in _MATRA_PHONES:
                units.append(("V", _MATRA_PHONES[nxt]))
                i += 1
            else:
                units.append(("S", SCHWA))  # inherent schwa, candidate for deletion
        elif ch in _VOWEL_PHONES:
            units.append(("V", _VOWEL_PHONES[ch]))
        elif ch in (ANUSVARA, CHANDRABINDU):
            units.append(("N", ch))
        elif ch == "ः":  # visarga: a breathy h
            units.append(("C", "h"))
        i += 1
    return units


def _delete_schwas(units: list[tuple[str, str]]) -> list[tuple[str, str]]:
    def is_vowel(u):
        return u[0] in ("V", "S")

    units = list(units)
    # Rule 1: word-final inherent schwa (only if the word has another vowel,
    # so single-letter words like "न" keep their vowel).
    for j in range(len(units) - 1, -1, -1):
        if units[j][0] == "N":
            continue
        if units[j][0] == "S" and sum(map(is_vowel, units)) > 1:
            units.pop(j)
        break
    # Rule 2: medial V C _ C V, right to left.
    j = len(units) - 1
    while j >= 0:
        if units[j][0] == "S":
            prev_c = j - 1 >= 0 and units[j - 1][0] == "C"
            prev_v = j - 2 >= 0 and is_vowel(units[j - 2])
            next_c = j + 1 < len(units) and units[j + 1][0] == "C"
            next_v = j + 2 < len(units) and is_vowel(units[j + 2])
            if prev_c and prev_v and next_c and next_v:
                units.pop(j)
        j -= 1
    return units


def word_to_phonemes(word: str) -> list[str]:
    units = _delete_schwas(_word_to_units(word))
    phones: list[str] = []
    for k, (kind, value) in enumerate(units):
        if kind == "N":
            # Anusvara before a consonant becomes the nasal of the same place;
            # otherwise (or chandrabindu) it nasalises the vowel.
            nxt = units[k + 1][1] if k + 1 < len(units) and units[k + 1][0] == "C" else None
            phones.append(_PLACE.get(nxt, "~") if value == ANUSVARA and nxt else "~")
        else:
            phones.append(value)
    return phones


def text_to_phonemes(text: str) -> list[list[str]]:
    """One phoneme list per word."""
    return [word_to_phonemes(w) for w in text.split()]


def phonemes_to_visemes(phones: list[str], merge_repeats: bool = True) -> list[str]:
    visemes = [VISEME_OF.get(p, "T") for p in phones]
    visemes = [v for v in visemes if v]
    if merge_repeats:
        # "pp" in अप्पा is one long closure on video, so repeats merge.
        visemes = [v for k, v in enumerate(visemes) if k == 0 or v != visemes[k - 1]]
    return visemes


def text_to_visemes(text: str, word_sep: str | None = "_") -> list[str]:
    """Viseme string for a sentence; ``word_sep`` marks word boundaries (None to drop)."""
    out: list[str] = []
    for k, word in enumerate(text.split()):
        if k and word_sep:
            out.append(word_sep)
        out.extend(phonemes_to_visemes(word_to_phonemes(word)))
    return out


# Letters that are visually confusable, used to brief the LLM corrector.
CONFUSABLE_LETTER_GROUPS = (
    "प फ ब भ म",
    "त थ द ध न ल स ट ठ ड ढ ण र ड़",
    "च छ ज झ श ष य",
    "क ख ग घ ह",
    "फ़ व",
)

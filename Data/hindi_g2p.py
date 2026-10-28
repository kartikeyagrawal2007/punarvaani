"""
Pure-Python Hindi Grapheme-to-Phoneme (G2P) Engine
Features:
- Devanagari Unicode NFC normalization
- Consonant and vowel extraction
- Matra, Anusvara (ं), Chandrabindu (ँ), Visarga (ः), and Nukta (़) processing
- 3-step Hindi Schwa Deletion (Schwa Vilopa) algorithm
- Zero external binary dependencies (no espeak-ng required)
"""

import unicodedata
import re
from typing import List

# ---------------------------------------------------------
# Devanagari Phoneme Maps
# ---------------------------------------------------------

# Independent Vowels
INDEPENDENT_VOWELS = {
    'अ': 'A',
    'आ': 'AA',
    'इ': 'I',
    'ई': 'II',
    'उ': 'U',
    'ऊ': 'UU',
    'ऋ': 'RI',
    'ए': 'E',
    'ऐ': 'AI',
    'ओ': 'O',
    'औ': 'AU',
}

# Dependent Vowel Signs (Matras)
VOWEL_MATRAS = {
    'ा': 'AA',
    'ि': 'I',
    'ी': 'II',
    'ु': 'U',
    'ू': 'UU',
    'ृ': 'RI',
    'े': 'E',
    'ै': 'AI',
    'ो': 'O',
    'ौ': 'AU',
}

# Consonants (each inherently contains schwa 'A' unless followed by halant or matra)
CONSONANTS = {
    # Velar
    'क': 'K', 'ख': 'KH', 'ग': 'G', 'घ': 'GH', 'ङ': 'NG',
    # Palatal
    'च': 'CH', 'छ': 'CHH', 'ज': 'J', 'झ': 'JH', 'ञ': 'NY',
    # Retroflex
    'ट': 'TR', 'ठ': 'THR', 'ड': 'DR', 'ढ': 'DHR', 'ण': 'NR',
    # Flaps (nukta on ड, ढ)
    'ड़': 'RD', 'ढ़': 'RHD',
    # Dental
    'त': 'TD', 'थ': 'THD', 'द': 'DD', 'ध': 'DHD', 'न': 'N',
    # Labial
    'प': 'P', 'फ': 'PH', 'ब': 'B', 'भ': 'BH', 'म': 'M',
    # Approximants / Semivowels
    'य': 'Y', 'र': 'R', 'ल': 'L', 'व': 'V',
    # Sibilants / Fricatives
    'श': 'SH', 'ष': 'SHR', 'स': 'S', 'ह': 'H',
    # Loan consonants (Nukta forms)
    'क़': 'Q', 'ख़': 'KH_LOAN', 'ग़': 'GH_LOAN', 'ज़': 'Z', 'फ़': 'F'
}

HALANT = '्'
ANUSVARA = 'ं'
CHANDRABINDU = 'ँ'
VISARGA = 'ः'
NUKTA = '़'

# Pre-compose common nukta characters to single standard forms
NUKTA_MAP = {
    'ड' + NUKTA: 'ड़',
    'ढ' + NUKTA: 'ढ़',
    'क' + NUKTA: 'क़',
    'ख' + NUKTA: 'ख़',
    'ग' + NUKTA: 'ग़',
    'ज' + NUKTA: 'ज़',
    'फ' + NUKTA: 'फ़',
}


def normalize_devanagari(text: str) -> str:
    """Normalize Unicode text and resolve Nuqta combinations."""
    text = unicodedata.normalize('NFC', text)
    for raw, merged in NUKTA_MAP.items():
        text = text.replace(raw, merged)
    return text


def word_to_syllables(word: str) -> List[dict]:
    """
    Decomposes a Devanagari word into a sequence of phonetic units (syllable structures).
    """
    word = normalize_devanagari(word)
    i = 0
    n = len(word)
    units = []

    while i < n:
        char = word[i]

        # Check for independent vowel
        if char in INDEPENDENT_VOWELS:
            vowel_token = INDEPENDENT_VOWELS[char]
            nasal = False
            if i + 1 < n and word[i + 1] in (ANUSVARA, CHANDRABINDU):
                nasal = True
                i += 1
            units.append({
                'type': 'vowel',
                'phoneme': vowel_token,
                'nasal': nasal
            })
            i += 1

        # Check for consonant (including 2-char nukta combinations like 'फ' + NUKTA)
        cons_token = None
        if i + 1 < n and (char + word[i + 1]) in CONSONANTS:
            cons_token = CONSONANTS[char + word[i + 1]]
            i += 2
        elif char in CONSONANTS:
            cons_token = CONSONANTS[char]
            i += 1

        if cons_token is not None:
            # Check next character modifier: halant, matra, or anusvara
            has_halant = False
            matra_token = None
            has_nasal = False

            if i < n and word[i] == HALANT:
                has_halant = True
                i += 1
            elif i < n and word[i] in VOWEL_MATRAS:
                matra_token = VOWEL_MATRAS[word[i]]
                i += 1
                if i < n and word[i] in (ANUSVARA, CHANDRABINDU):
                    has_nasal = True
                    i += 1
            elif i < n and word[i] in (ANUSVARA, CHANDRABINDU):
                has_nasal = True
                i += 1

            units.append({
                'type': 'consonant',
                'phoneme': cons_token,
                'has_schwa': (not has_halant and matra_token is None),
                'matra': matra_token,
                'has_halant': has_halant,
                'has_nasal': has_nasal
            })


        elif char in (ANUSVARA, CHANDRABINDU):
            if units:
                units[-1]['has_nasal'] = True
            i += 1
        elif char == VISARGA:
            units.append({'type': 'visarga', 'phoneme': 'AH'})
            i += 1
        else:
            # Ignore punctuation, spaces, or unrecognized symbols
            i += 1

    return units


def apply_schwa_deletion(units: List[dict]) -> List[str]:
    """
    Applies standard Hindi Schwa Deletion (Choudhury & Rao rule):
    1. Word-final schwa is deleted (unless word is a monosyllabic single consonant or ends in conjunct).
    2. Medial schwa in V-C-schwa-C-V context is deleted.
    """
    num_units = len(units)
    if num_units == 0:
        return []

    # Count how many syllables/units have vowels or inherent schwas
    vocalic_count = 0
    for u in units:
        if u['type'] == 'vowel' or u.get('matra') is not None or u.get('has_schwa', False):
            vocalic_count += 1

    # Rule 1: Final schwa deletion
    # In spoken Hindi, word-final schwa is deleted for words with >= 2 units
    if num_units >= 2:
        last_unit = units[-1]
        if last_unit['type'] == 'consonant' and last_unit.get('has_schwa', False):
            last_unit['has_schwa'] = False


    # Rule 2: Medial schwa deletion
    # For a pattern: Unit[i-1] is vocalic, Unit[i] has schwa, Unit[i+1] is followed by a vowel
    if num_units >= 3:
        for i in range(1, num_units - 1):
            curr = units[i]
            if curr['type'] == 'consonant' and curr.get('has_schwa', False):
                # Check if preceding unit has vowel/matra
                prev_vocalic = (units[i - 1]['type'] == 'vowel' or 
                                units[i - 1].get('matra') is not None or 
                                units[i - 1].get('has_schwa', False))
                # Check if succeeding unit has explicit vowel/matra
                next_vocalic = (units[i + 1]['type'] == 'vowel' or 
                                units[i + 1].get('matra') is not None)

                if prev_vocalic and next_vocalic and not units[i - 1].get('has_halant', False):
                    curr['has_schwa'] = False

    # Convert units to phoneme list
    phonemes = []
    for u in units:
        if u['type'] == 'vowel':
            phonemes.append(u['phoneme'])
            if u.get('nasal', False):
                phonemes.append('AN')
        elif u['type'] == 'consonant':
            phonemes.append(u['phoneme'])
            if u.get('has_schwa', False):
                phonemes.append('A')
            elif u.get('matra') is not None:
                phonemes.append(u['matra'])
            if u.get('has_nasal', False):
                phonemes.append('AN')
        elif u['type'] == 'visarga':
            phonemes.append('AH')

    return phonemes


def devanagari_to_phonemes(text: str, as_string: bool = False):
    """
    Converts Hindi Devanagari sentence to phonemes with word boundary tokens '|'.

    Example:
        'नमस्ते भारत' -> ['N', 'M', 'S', 'TD', 'E', '|', 'BH', 'AA', 'R', 'TD']
    """
    words = text.strip().split()
    all_phonemes = []

    for idx, word in enumerate(words):
        # Strip punctuation
        cleaned = re.sub(r'[^\u0900-\u097F]', '', word)
        if not cleaned:
            continue
        units = word_to_syllables(cleaned)
        word_phonemes = apply_schwa_deletion(units)
        if word_phonemes:
            if all_phonemes:
                all_phonemes.append('|')  # Word boundary delimiter
            all_phonemes.extend(word_phonemes)

    if as_string:
        return ' '.join(all_phonemes)
    return all_phonemes


if __name__ == '__main__':
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    test_cases = [
        ("नमस्ते", "N M S TD E"),
        ("भारत", "BH AA R TD"),
        ("कमल", "K M L"),
        ("लड़का", "L RD K AA"),
        ("फ़िल्म", "F I L M"),
        ("यह एक परीक्षण है", "Y H | E K | P R II K SH NR | H AI")
    ]
    print("Testing Hindi G2P:")
    for text, _ in test_cases:
        phon = devanagari_to_phonemes(text, as_string=True)
        print(f"'{text}' -> {phon}")


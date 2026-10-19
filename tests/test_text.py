from punarvaani.text.hindi import is_mostly_devanagari, normalize, number_to_words
from punarvaani.text.phonemes import text_to_visemes, word_to_phonemes
from punarvaani.text.vocab import CharVocab


def test_nukta_forms_normalise_to_one_spelling():
    precomposed = "ज़"  # ज़ as one code point
    decomposed = "ज़"  # ज + nukta
    assert normalize(precomposed) == normalize(decomposed) == decomposed
    assert normalize("ज़रूर", level="loose") == "जरूर"


def test_punctuation_digits_and_invisibles():
    assert normalize("नमस्ते!  आप कैसे हैं?।") == "नमस्ते आप कैसे हैं"
    assert normalize("मुझे २ गोली दो") == "मुझे दो गोली दो"
    assert normalize("कमरा 21") == "कमरा इक्कीस"
    assert normalize("क्‍ष") == "क्ष"


def test_number_to_words_indian_system():
    assert number_to_words(0) == "शून्य"
    assert number_to_words(99) == "निन्यानवे"
    assert number_to_words(105) == "एक सौ पाँच"
    assert number_to_words(2024) == "दो हज़ार चौबीस"
    assert number_to_words(15_00_000) == "पंद्रह लाख"
    assert number_to_words(3_00_00_000) == "तीन करोड़"


def test_code_mixing_detector():
    assert is_mostly_devanagari("मुझे पानी चाहिए")
    assert not is_mostly_devanagari("मुझे doctor को call करना है")


def test_vocab_roundtrip():
    vocab = CharVocab()
    ids = vocab.encode("मुझे पानी चाहिए")
    assert vocab.blank_id == 0 and vocab.unk_id not in ids
    assert vocab.decode(ids) == "मुझे पानी चाहिए"


def test_schwa_deletion():
    assert word_to_phonemes("कमल") == ["k", "a", "m", "a", "l"]  # final schwa only
    assert word_to_phonemes("कमला") == ["k", "a", "m", "l", "aa"]  # medial VC_CV
    assert word_to_phonemes("धड़कन") == ["dh", "a", "R", "k", "a", "n"]
    assert word_to_phonemes("नमस्ते") == ["n", "a", "m", "a", "s", "t", "e"]  # cluster blocks deletion
    assert word_to_phonemes("न") == ["n", "a"]  # single letter keeps its vowel


def test_anusvara_is_homorganic():
    assert word_to_phonemes("अंक") == ["a", "ng", "k"]
    assert word_to_phonemes("हिंदी") == ["h", "i", "n", "d", "ii"]
    assert word_to_phonemes("संबंध")[:3] == ["s", "a", "m"]


def test_pa_ba_ma_are_one_viseme():
    assert text_to_visemes("पल") == text_to_visemes("बल") == text_to_visemes("मल") == ["P", "A", "T"]
    assert text_to_visemes("पानी") != text_to_visemes("कानी")

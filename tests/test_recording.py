from pathlib import Path

from punarvaani.models.path_a.phrase_mode import PhraseBank
from punarvaani.recording.prompts import (
    coverage,
    diphones,
    load_phrases,
    plan_session,
    select_balanced_sentences,
)
from punarvaani.text.vocab import CharVocab

PHRASES = Path(__file__).resolve().parents[1] / "configs" / "phrases_hi.txt"


def test_phrase_file_loads_into_a_phrase_bank():
    phrases = load_phrases(PHRASES)
    assert len(phrases) >= 150
    bank = PhraseBank(phrases, CharVocab())  # raises on duplicates after normalisation
    assert len(bank) == len(phrases)


def test_greedy_selection_beats_random_coverage():
    corpus = load_phrases(PHRASES)
    chosen = select_balanced_sentences(corpus, 15)
    assert len(chosen) == 15
    first15 = corpus[:15]
    assert coverage(chosen, corpus) > coverage(first15, corpus)
    assert diphones("पानी") == {("#", "p"), ("p", "aa"), ("aa", "n"), ("n", "ii"), ("ii", "#")}


def test_session_plan_pairs_spoken_and_silent():
    plan = plan_session(["मुझे पानी चाहिए", "धन्यवाद"], ["आज मौसम अच्छा है"])
    assert len(plan) == 6
    for a, b in zip(plan[::2], plan[1::2]):
        assert a.text == b.text and (a.mode, b.mode) == ("spoken", "silent")

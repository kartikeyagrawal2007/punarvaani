"""Golden-table tests for Hindi G2P and schwa deletion."""

import unittest

from Data.hindi_g2p import (
    clear_unsupported_char_log,
    devanagari_to_phonemes,
    get_unsupported_char_report,
)
from config import get_hindi_vocab


class TestHindiG2PGoldenTable(unittest.TestCase):
    GOLDEN = {
        "नमस्ते": "N A M A S TD E",
        "भारत": "BH AA R A TD",
        "कमल": "K A M A L",
        "कमला": "K A M L AA",
        "लड़का": "L A RD K AA",
        "फ़िल्म": "F I L M",
        "यह": "Y A H",
        "विद्वान": "V I DD V AA N",
        "हिंदी": "H I AN DD II",
        "अतः": "A TD AH",
        "क्या": "K Y AA",
        "आइए": "AA I E",
        "समझना": "S A M A JH N AA",
        "मिलना": "M I L N AA",
        "अमर": "A M A R",
        "कर्म": "K A R M",
        "डॉक्टर": "DR O K TR A R",
    }

    def test_golden_transcriptions(self):
        for word, expected in self.GOLDEN.items():
            with self.subTest(word=word):
                self.assertEqual(devanagari_to_phonemes(word, as_string=True), expected)

    def test_multiword_sentence(self):
        self.assertEqual(
            devanagari_to_phonemes("यह एक परीक्षण है", as_string=True),
            "Y A H | E K | P A R II K SHR A NR | H AI",
        )

    def test_every_emitted_token_is_in_vocabulary(self):
        vocab = get_hindi_vocab()
        for word in self.GOLDEN:
            for token in devanagari_to_phonemes(word):
                self.assertIn(token, vocab, f"{token} from '{word}' missing from HPI")

    def test_unsupported_characters_are_logged(self):
        clear_unsupported_char_log()
        devanagari_to_phonemes("abc 123", transcript_id="clip_7")
        report = get_unsupported_char_report()
        self.assertGreater(report["total_unsupported_occurrences"], 0)
        self.assertIn("clip_7", report["affected_transcript_ids"])


if __name__ == "__main__":
    unittest.main()

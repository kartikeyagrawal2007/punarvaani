"""Sarvam adapter validation and explicit MOCK-mode reasons."""

import json
import os
import tempfile
import unittest

from Models.Sarvam import (
    TAGS,
    HindiPhonemeTranslator,
    validate_sarvam_adapter_compatibility,
)


class TestSarvamLoader(unittest.TestCase):
    def test_no_adapter_reports_mock_reason(self):
        translator = HindiPhonemeTranslator(adapter_path=None, device="cpu")
        self.assertTrue(translator.is_mock)
        self.assertIn("no adapter", translator.mock_reason)
        self.assertIn("MOCK TRANSCRIPTION", translator.translate("N M S TD E"))

    def test_missing_adapter_path_reports_reason(self):
        translator = HindiPhonemeTranslator(adapter_path="does/not/exist", device="cpu")
        self.assertTrue(translator.is_mock)
        self.assertIn("not found", translator.mock_reason)

    def test_prompt_matches_contract(self):
        translator = HindiPhonemeTranslator(adapter_path=None, device="cpu")
        prompt = translator.build_prompt("N M S TD E")
        self.assertEqual(
            prompt,
            "<S2S>\n<PHONEMES>\nN M S TD E\n</PHONEMES>\n<TEXT>\n",
        )
        for tag in TAGS:
            self.assertIn(tag, prompt)

    def test_mismatched_base_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "adapter_config.json"), "w", encoding="utf-8") as handle:
                json.dump({"base_model_name_or_path": "some/other-model"}, handle)
            with self.assertRaises(ValueError):
                validate_sarvam_adapter_compatibility(
                    tmp, base_model_id="sarvamai/sarvam-1", strict=True
                )

    def test_missing_adapter_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                validate_sarvam_adapter_compatibility(
                    tmp, base_model_id="sarvamai/sarvam-1", strict=True
                )

    def test_matching_config_passes_without_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "adapter_config.json"), "w", encoding="utf-8") as handle:
                json.dump({"base_model_name_or_path": "sarvamai/sarvam-1"}, handle)
            report = validate_sarvam_adapter_compatibility(
                tmp, base_model_id="sarvamai/sarvam-1", strict=True
            )
            self.assertTrue(report["is_compatible"])


if __name__ == "__main__":
    unittest.main()

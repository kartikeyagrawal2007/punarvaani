"""Preflight dataset and adapter validation."""

import json
import os
import shutil
import tempfile
import unittest

from Data.synthetic import create_synthetic_dataset
from scripts.preflight import check_adapter, main as preflight_main, scan_dataset


class TestPreflightDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vallr_preflight_")
        create_synthetic_dataset(
            self.tmp, speakers=2, clips=1, seconds=0.6, fps=25, size=(48, 48)
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_clean_dataset_passes(self):
        report = scan_dataset(self.tmp)
        self.assertEqual(report["blocking"], [])
        self.assertEqual(report["pairs"], 2)
        self.assertEqual(len(report["speakers"]), 2)
        self.assertEqual(report["invalid_pct"], 0.0)

    def test_missing_transcript_blocks(self):
        os.remove(os.path.join(self.tmp, "spk01", "spk01_001.txt"))
        report = scan_dataset(self.tmp)
        self.assertTrue(report["missing_transcripts"])
        self.assertTrue(any("missing transcript" in issue for issue in report["blocking"]))

    def test_root_level_clip_blocks_without_flag(self):
        root_dir = tempfile.mkdtemp(prefix="vallr_preflight_root_")
        try:
            create_synthetic_dataset(root_dir, speakers=1, clips=1, seconds=0.4, size=(32, 32))
            inner = os.path.join(root_dir, "spk01")
            for name in os.listdir(inner):
                shutil.move(os.path.join(inner, name), os.path.join(root_dir, name))
            os.rmdir(inner)

            report = scan_dataset(root_dir)
            self.assertTrue(report["unknown_speaker"])
            self.assertTrue(any("speaker folder" in issue for issue in report["blocking"]))

            report = scan_dataset(root_dir, allow_unknown_speakers=True)
            self.assertEqual(report["blocking"], [])
        finally:
            shutil.rmtree(root_dir, ignore_errors=True)

    def test_main_exit_codes(self):
        output = os.path.join(self.tmp, "report.json")
        self.assertEqual(preflight_main(["--data_dir", self.tmp, "--output", output]), 0)
        self.assertTrue(os.path.exists(output))

        os.remove(os.path.join(self.tmp, "spk02", "spk02_001.txt"))
        self.assertEqual(preflight_main(["--data_dir", self.tmp, "--output", output]), 1)


class TestPreflightAdapter(unittest.TestCase):
    def _make_adapter(self, base_model="sarvamai/sarvam-1", with_weights=True):
        directory = tempfile.mkdtemp(prefix="vallr_adapter_")
        with open(os.path.join(directory, "adapter_config.json"), "w", encoding="utf-8") as handle:
            json.dump({"base_model_name_or_path": base_model}, handle)
        if with_weights:
            with open(os.path.join(directory, "adapter_model.safetensors"), "wb") as handle:
                handle.write(b"")
        return directory

    def test_valid_adapter(self):
        directory = self._make_adapter()
        try:
            report = check_adapter(directory)
            self.assertTrue(report["compatible"])
            self.assertEqual(report["blocking"], [])
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_missing_weights_blocks(self):
        directory = self._make_adapter(with_weights=False)
        try:
            report = check_adapter(directory)
            self.assertFalse(report["compatible"])
            self.assertTrue(report["blocking"])
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_wrong_base_model_blocks(self):
        directory = self._make_adapter(base_model="meta-llama/Llama-2-7b")
        try:
            report = check_adapter(directory)
            self.assertFalse(report["compatible"])
            self.assertTrue(any("base model" in issue for issue in report["blocking"]))
        finally:
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

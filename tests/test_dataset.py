"""Dataset discovery, speaker-disjoint splits, all-frames sampling and CTC precheck."""

import os
import shutil
import tempfile
import unittest

from Data.dataset import HAS_FACE_CROPPER, HindiVideoDataset
from Data.synthetic import create_synthetic_dataset
from config import get_hindi_vocab


class TestHindiVideoDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocab = get_hindi_vocab()

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vallr_dataset_")
        create_synthetic_dataset(
            self.tmp, speakers=3, clips=2, seconds=0.6, fps=25, size=(48, 48)
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_dataset(self, split="train", **overrides):
        params = dict(
            data_dir=self.tmp,
            phoneme_vocab=self.vocab,
            split=split,
            val_split=0.34,
            crop_mode="none",
            max_frames=96,
            num_frames=0,
            validate_ctc=True,
            tubelet_size=2,
            temporal_upsample_factor=4,
        )
        params.update(overrides)
        return HindiVideoDataset(**params)

    def test_split_is_speaker_disjoint(self):
        train = self.make_dataset("train")
        val = self.make_dataset("val")

        train_speakers = {sample["speaker"] for sample in train.samples}
        val_speakers = {sample["speaker"] for sample in val.samples}
        self.assertTrue(train_speakers)
        self.assertTrue(val_speakers)
        self.assertFalse(train_speakers.intersection(val_speakers))
        self.assertEqual(train_speakers | val_speakers, {"spk01", "spk02", "spk03"})

    def test_split_is_deterministic(self):
        first = self.make_dataset("train")
        second = self.make_dataset("train")
        self.assertEqual(
            [s["video"] for s in first.samples], [s["video"] for s in second.samples]
        )

    def test_dedicated_split_uses_everything(self):
        dataset = self.make_dataset("val", is_dedicated_split=True)
        self.assertEqual(len(dataset), 6)

    def test_missing_transcript_is_skipped(self):
        missing = os.path.join(self.tmp, "spk01", "spk01_001.txt")
        os.remove(missing)
        dataset = self.make_dataset("train", is_dedicated_split=True)
        self.assertEqual(len(dataset), 5)

    def test_root_level_clips_require_flag(self):
        root_dir = tempfile.mkdtemp(prefix="vallr_root_")
        try:
            create_synthetic_dataset(root_dir, speakers=1, clips=1, seconds=0.4, size=(32, 32))
            inner = os.path.join(root_dir, "spk01")
            for name in os.listdir(inner):
                shutil.move(os.path.join(inner, name), os.path.join(root_dir, name))
            os.rmdir(inner)

            with self.assertRaises(ValueError):
                HindiVideoDataset(
                    data_dir=root_dir,
                    phoneme_vocab=self.vocab,
                    val_split=0.0,
                    crop_mode="none",
                    validate_ctc=False,
                )

            dataset = HindiVideoDataset(
                data_dir=root_dir,
                phoneme_vocab=self.vocab,
                val_split=0.0,
                crop_mode="none",
                validate_ctc=False,
                allow_unknown_speakers=True,
            )
            self.assertEqual(len(dataset), 1)
        finally:
            shutil.rmtree(root_dir, ignore_errors=True)

    def test_single_speaker_cannot_make_disjoint_split(self):
        single = tempfile.mkdtemp(prefix="vallr_single_")
        try:
            create_synthetic_dataset(single, speakers=1, clips=2, seconds=0.4, size=(32, 32))
            with self.assertRaises(ValueError):
                HindiVideoDataset(
                    data_dir=single,
                    phoneme_vocab=self.vocab,
                    split="train",
                    val_split=0.5,
                    crop_mode="none",
                    validate_ctc=False,
                )
        finally:
            shutil.rmtree(single, ignore_errors=True)

    def test_all_frames_are_loaded(self):
        dataset = self.make_dataset("train", is_dedicated_split=True)
        item = dataset[0]
        self.assertIsNotNone(item)
        self.assertEqual(item.num_frames, item.video.size(0))
        self.assertGreaterEqual(item.num_frames, 10)
        self.assertLessEqual(item.num_frames, 96)
        self.assertIsInstance(item.speaker_id, str)
        self.assertGreater(len(item.phoneme_ids), 0)

    def test_strict_ctc_precheck_raises(self):
        with self.assertRaises(RuntimeError):
            self.make_dataset(
                "train",
                is_dedicated_split=True,
                max_frames=4,
                max_invalid_sample_pct=0.0,
                strict_ctc=True,
            )

    @unittest.skipIf(HAS_FACE_CROPPER, "mediapipe is installed in this environment")
    def test_mouth_crop_without_mediapipe_fails_fast(self):
        with self.assertRaises(RuntimeError):
            self.make_dataset(
                "train", is_dedicated_split=True, crop_mode="mouth"
            )

    def test_invalid_fraction_is_reported(self):
        dataset = self.make_dataset(
            "train",
            is_dedicated_split=True,
            max_frames=4,
            strict_ctc=False,
        )
        self.assertGreater(dataset.invalid_pct, 0.0)


if __name__ == "__main__":
    unittest.main()

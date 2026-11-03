"""Offline preprocessing: mouth-crop call, relative paths, report."""

import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from Data.synthetic import create_synthetic_dataset
from scripts.preprocess_lips import preprocess_dataset


class FakeCropper:
    def __init__(self, fail=False):
        self.fail = fail
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def crop_mouth(self, image, frame_size=None):
        if self.fail:
            return None
        if frame_size is None:
            return image
        if isinstance(frame_size, (tuple, list)):
            size = (int(frame_size[0]), int(frame_size[1]))
        else:
            size = (int(frame_size), int(frame_size))
        return cv2.resize(image, size)


class TestPreprocessLips(unittest.TestCase):
    def setUp(self):
        self.input_dir = tempfile.mkdtemp(prefix="vallr_pre_in_")
        self.output_dir = tempfile.mkdtemp(prefix="vallr_pre_out_")
        create_synthetic_dataset(
            self.input_dir, speakers=2, clips=1, seconds=0.4, fps=25, size=(64, 64)
        )

    def tearDown(self):
        shutil.rmtree(self.input_dir, ignore_errors=True)
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_crops_and_preserves_speaker_tree(self):
        cropper = FakeCropper()
        report = preprocess_dataset(
            self.input_dir,
            self.output_dir,
            max_frames=4,
            frame_size=32,
            cropper=cropper,
        )

        self.assertEqual(report["successful_crops"], 2)
        self.assertEqual(report["crop_success_rate_pct"], 100.0)
        self.assertEqual(cropper.reset_calls, 2)

        for speaker in ("spk01", "spk02"):
            video = os.path.join(self.output_dir, speaker, f"{speaker}_001.mp4")
            transcript = os.path.join(self.output_dir, speaker, f"{speaker}_001.txt")
            self.assertTrue(os.path.exists(video))
            self.assertTrue(os.path.exists(transcript))
            cap = cv2.VideoCapture(video)
            try:
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 32)
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 32)
            finally:
                cap.release()

        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "preprocessing_report.json")))

    def test_failed_crops_are_reported(self):
        report = preprocess_dataset(
            self.input_dir,
            self.output_dir,
            max_frames=4,
            frame_size=32,
            cropper=FakeCropper(fail=True),
        )
        self.assertEqual(report["successful_crops"], 0)
        self.assertEqual(len(report["failed_videos"]), 2)
        self.assertTrue(all("mouth crop failed" in item["reason"] for item in report["failed_videos"]))


if __name__ == "__main__":
    unittest.main()

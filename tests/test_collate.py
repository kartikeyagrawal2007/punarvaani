"""Variable-length padding and CTC length accounting."""

import math
import unittest

import torch

from Data.dataset import VSRItem, custom_collate


def make_item(num_frames, ids, text="टेस्ट"):
    return VSRItem(
        video=torch.rand(num_frames, 3, 8, 8),
        phoneme_ids=torch.tensor(ids, dtype=torch.long),
        text=text,
        speaker_id="spk01",
        num_frames=num_frames,
    )


class TestCustomCollate(unittest.TestCase):
    def test_pads_to_max_frames_and_sets_lengths(self):
        items = [make_item(6, [2, 3]), make_item(10, [4, 5, 6])]
        batch = custom_collate(items, tubelet_size=2, temporal_upsample_factor=4)

        self.assertEqual(batch.videos.shape, (2, 10, 3, 8, 8))
        self.assertEqual(batch.video_lengths.tolist(), [6, 10])
        self.assertEqual(
            batch.input_lengths.tolist(),
            [math.ceil(6 / 2) * 4, math.ceil(10 / 2) * 4],
        )
        self.assertEqual(batch.dropped, 0)
        # padded frames are exactly zero
        self.assertEqual(float(batch.videos[0, 6:].abs().sum()), 0.0)
        self.assertGreater(float(batch.videos[1, 5:].abs().sum()), 0.0)

    def test_drops_samples_that_violate_ctc_constraint(self):
        good = make_item(20, [2, 3])
        impossible = make_item(2, list(range(2, 40)))  # U=38 > T_ctc=4
        batch = custom_collate([good, impossible], tubelet_size=2, temporal_upsample_factor=4)
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch.dropped, 1)

    def test_all_invalid_returns_empty_batch(self):
        impossible = make_item(2, list(range(2, 40)))
        batch = custom_collate([impossible, impossible], tubelet_size=2, temporal_upsample_factor=4)
        self.assertIsNone(batch.videos)
        self.assertEqual(batch.dropped, 2)
        self.assertEqual(len(batch), 0)

    def test_legacy_tuple_and_fixed_t_ctc(self):
        label = torch.tensor([15, 15], dtype=torch.long)
        sample = (torch.zeros(16, 3, 224, 224), label, "टेस्ट")
        batch = custom_collate([sample], t_ctc=3)
        self.assertIsNotNone(batch.videos)
        self.assertEqual(batch.input_lengths.tolist(), [3])
        self.assertEqual(batch.video_lengths.tolist(), [16])

    def test_none_items_are_ignored(self):
        batch = custom_collate([None, make_item(4, [2])], tubelet_size=2, temporal_upsample_factor=4)
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch.dropped, 0)


if __name__ == "__main__":
    unittest.main()

"""Variable-length forward pass, CTC lengths and padding invariance."""

import unittest

import torch

from Models.ML_VALLR import ML_VALLR


def tiny_model(temporal_upsample_factor=2):
    return ML_VALLR(
        adapter_dim=16,
        num_classes=10,
        feature_size=32,
        img_size=64,
        patch_size=16,
        tubelet_size=2,
        vit_depth=1,
        vit_heads=4,
        vit_dropout=0.0,
        vit_attn_dropout=0.0,
        temporal_upsample_factor=temporal_upsample_factor,
    ).eval()


class TestVariableLengthModel(unittest.TestCase):
    def test_batched_lengths_and_logit_shape(self):
        model = tiny_model()
        videos = torch.randn(2, 3, 8, 64, 64)
        lengths = torch.tensor([8, 6])
        logits, adapted = model(videos, video_lengths=lengths)
        self.assertEqual(logits.shape, (2, 8, 10))
        self.assertEqual(adapted.shape, (2, 8, 256))

        expected = model.compute_input_lengths(lengths, tubelet_size=2, temporal_upsample_factor=2)
        self.assertEqual(expected.tolist(), [8, 6])
        self.assertLessEqual(int(expected.max()), logits.size(1))

    def test_no_lengths_uses_full_clip(self):
        model = tiny_model()
        logits, _ = model(torch.randn(1, 3, 8, 64, 64))
        self.assertEqual(logits.shape, (1, 8, 10))

    def test_accepts_time_first_layout(self):
        model = tiny_model()
        logits, _ = model(torch.randn(1, 8, 3, 64, 64))
        self.assertEqual(logits.shape, (1, 8, 10))

    def test_padding_does_not_change_real_frames(self):
        torch.manual_seed(0)
        model = tiny_model()
        short = torch.randn(1, 3, 6, 64, 64)
        padded = torch.zeros(1, 3, 8, 64, 64)
        padded[:, :, :6] = short

        short_logits, _ = model(short, video_lengths=torch.tensor([6]))
        padded_logits, _ = model(padded, video_lengths=torch.tensor([6]))

        # 6 frames -> 3 tubelets -> 6 CTC steps
        torch.testing.assert_close(
            short_logits[:, :6], padded_logits[:, :6], atol=1e-5, rtol=1e-4
        )

    def test_padding_invariance_with_spatial_token_pooling(self):
        torch.manual_seed(1)
        model = ML_VALLR(
            adapter_dim=16,
            num_classes=10,
            feature_size=32,
            img_size=64,
            patch_size=16,
            tubelet_size=2,
            vit_depth=1,
            vit_heads=4,
            vit_dropout=0.0,
            temporal_upsample_factor=2,
            spatial_pool_tokens=(2, 2),
        ).eval()
        short = torch.randn(1, 3, 6, 64, 64)
        padded = torch.zeros(1, 3, 8, 64, 64)
        padded[:, :, :6] = short

        short_logits, _ = model(short, video_lengths=torch.tensor([6]))
        padded_logits, _ = model(padded, video_lengths=torch.tensor([6]))
        torch.testing.assert_close(
            short_logits[:, :6], padded_logits[:, :6], atol=1e-5, rtol=1e-4
        )

    def test_input_lengths_never_exceed_available_steps(self):
        lengths = torch.tensor([1, 2, 5, 8])
        steps = ML_VALLR.compute_input_lengths(lengths, tubelet_size=2, temporal_upsample_factor=4)
        self.assertEqual(steps.tolist(), [4, 4, 12, 16])


if __name__ == "__main__":
    unittest.main()

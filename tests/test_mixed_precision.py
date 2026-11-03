"""CTC must stay finite under autocast (fp32 log-softmax)."""

import unittest

import torch
import torch.nn as nn

from main import _amp_context, _ctc_loss
from Models.ML_VALLR import ML_VALLR


class TestMixedPrecisionCTC(unittest.TestCase):
    def _tiny(self):
        return ML_VALLR(
            adapter_dim=16,
            num_classes=12,
            feature_size=32,
            img_size=64,
            patch_size=16,
            tubelet_size=2,
            vit_depth=1,
            vit_heads=4,
            vit_dropout=0.0,
            temporal_upsample_factor=2,
        )

    def test_cpu_amp_uses_no_scaler(self):
        context, scaler = _amp_context(torch.device("cpu"))
        self.assertIsNone(scaler)
        with context:
            logits = torch.randn(2, 8, 12)
            self.assertTrue(torch.isfinite(logits).all())

    def test_ctc_loss_is_finite_under_autocast(self):
        device = torch.device("cpu")
        model = self._tiny()
        criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        videos = torch.randn(2, 3, 8, 64, 64)
        labels = [torch.tensor([2, 3, 4]), torch.tensor([5, 6])]

        context, _ = _amp_context(device)
        with context:
            logits, _ = model(videos, video_lengths=torch.tensor([8, 6]))
            loss = _ctc_loss(criterion, logits, labels, torch.tensor([8, 6]), device)

        self.assertTrue(torch.isfinite(loss), f"CTC loss not finite: {loss}")
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_fp32_cast_is_applied(self):
        # float() before log_softmax must make the result fp32 even if logits are bf16.
        logits = torch.randn(1, 8, 12, dtype=torch.bfloat16)
        log_probs = logits.float().log_softmax(dim=-1)
        self.assertEqual(log_probs.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()

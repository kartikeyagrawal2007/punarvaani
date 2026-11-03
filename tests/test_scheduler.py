"""Warmup scheduler must warm up linearly, then delegate to the plateau scheduler."""

import unittest

import torch

from config import WarmupScheduler


class RecordingScheduler:
    def __init__(self):
        self.calls = []
    def step(self, loss=None):
        self.calls.append(loss)


class TestWarmupScheduler(unittest.TestCase):
    def test_linear_warmup_then_delegation(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        after = RecordingScheduler()
        scheduler = WarmupScheduler(
            optimizer,
            warmup_steps=2,
            base_lr=0.0,
            target_lr=1.0,
            after_scheduler=after,
        )

        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.5)
        self.assertEqual(after.calls, [])

        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.0)
        self.assertEqual(after.calls, [])

        scheduler.step(loss=0.7)
        self.assertEqual(after.calls, [0.7])

        scheduler.step()
        self.assertEqual(after.calls, [0.7, None])

    def test_last_lr_is_reported(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scheduler = WarmupScheduler(optimizer, warmup_steps=1, base_lr=0.0, target_lr=0.2)
        scheduler.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 0.2)


if __name__ == "__main__":
    unittest.main()

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

    def test_warmup_counts_batches_and_plateau_waits_for_it(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        after = RecordingScheduler()
        scheduler = WarmupScheduler(optimizer, warmup_steps=4, base_lr=0.0, target_lr=1e-4, after_scheduler=after)

        for _ in range(2):  # two batches of the first epoch
            scheduler.step_batch()
        scheduler.step_epoch(0.9)  # an epoch ending mid-warmup must not touch the plateau scheduler
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5e-5)
        self.assertEqual(after.calls, [])

        for _ in range(10):  # warmup ends after 4 updates; later batches leave the LR alone
            scheduler.step_batch()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-4)
        scheduler.step_epoch(0.8)
        self.assertEqual(after.calls, [0.8])

    def test_last_lr_is_reported(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scheduler = WarmupScheduler(optimizer, warmup_steps=1, base_lr=0.0, target_lr=0.2)
        scheduler.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 0.2)

    def test_state_round_trip_resumes_warmup_and_plateau(self):
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        def make():
            parameter = torch.nn.Parameter(torch.zeros(1))
            optimizer = torch.optim.SGD([parameter], lr=1.0)
            after = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-6)
            return optimizer, WarmupScheduler(optimizer, warmup_steps=4, base_lr=0.0, target_lr=1e-3,
                                              after_scheduler=after)

        optimizer, scheduler = make()
        for _ in range(2):
            scheduler.step_batch()
        state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}

        optimizer2, scheduler2 = make()
        optimizer2.load_state_dict(state["optimizer"])
        scheduler2.load_state_dict(state["scheduler"])
        self.assertEqual(scheduler2.current_step, 2)
        self.assertAlmostEqual(optimizer2.param_groups[0]["lr"], 5e-4)
        self.assertAlmostEqual(scheduler2.get_last_lr()[0], 5e-4)
        for _ in range(2):  # warmup continues where it stopped instead of starting again
            scheduler2.step_batch()
        self.assertAlmostEqual(optimizer2.param_groups[0]["lr"], 1e-3)

        for loss in (1.0, 1.0, 1.0):  # plateau patience state survives a second round trip
            scheduler2.step_epoch(loss)
        optimizer3, scheduler3 = make()
        optimizer3.load_state_dict(optimizer2.state_dict())
        scheduler3.load_state_dict(scheduler2.state_dict())
        self.assertEqual(scheduler3.after_scheduler.num_bad_epochs, scheduler2.after_scheduler.num_bad_epochs)
        self.assertAlmostEqual(optimizer3.param_groups[0]["lr"], optimizer2.param_groups[0]["lr"])


if __name__ == "__main__":
    unittest.main()

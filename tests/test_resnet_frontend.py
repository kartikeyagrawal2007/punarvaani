import unittest
from types import SimpleNamespace

import torch

from Models.ML_VALLR import VALLR, stage1_kwargs_from_state_dict


def _tiny(**kw):
    torch.manual_seed(0)
    return VALLR(num_classes=12, frontend="resnet", backend_dim=64, backend_layers=1, **kw).eval()


class TestResNetFrontend(unittest.TestCase):
    def test_ctc_length_matches_dataset_formula(self):
        model = _tiny()
        for frames in (5, 8, 13):
            with torch.no_grad():
                logits, _ = model(torch.rand(1, frames, 3, 64, 64), torch.tensor([frames]))
            self.assertEqual(logits.shape[1], -(-frames // 2) * 4)  # ceil(frames / tubelet 2) * upsample 4

    def test_padding_does_not_change_a_clip(self):
        model = _tiny()
        short, long_ = torch.rand(1, 7, 3, 64, 64), torch.rand(1, 12, 3, 64, 64)
        batch = torch.zeros(2, 12, 3, 64, 64)
        batch[0, :7], batch[1] = short[0], long_[0]
        with torch.no_grad():
            alone, _ = model(short, torch.tensor([7]))
            padded, _ = model(batch, torch.tensor([7, 12]))
        self.assertLess((alone[0] - padded[0, : alone.size(1)]).abs().max().item(), 1e-4)

    def test_checkpoint_rebuilds_the_same_architecture(self):
        state = _tiny().state_dict()
        kwargs = stage1_kwargs_from_state_dict(state)
        self.assertEqual(kwargs, {"frontend": "resnet", "backend_dim": 64, "backend_layers": 1})
        VALLR(num_classes=12, **kwargs).load_state_dict(state)
        vit = VALLR(num_classes=12, feature_size=32, vit_depth=1, vit_heads=4).state_dict()
        self.assertEqual(stage1_kwargs_from_state_dict(vit), {"frontend": "vit"})

    def test_training_step_reduces_loss_on_a_fixed_batch(self):
        model = _tiny()
        model.train()
        video = torch.rand(2, 10, 3, 64, 64)
        lengths = torch.tensor([10, 8])
        targets = torch.tensor([1, 2, 3, 4, 5, 6])
        ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = []
        for _ in range(15):
            logits, _ = model(video, lengths)
            loss = ctc(logits.log_softmax(-1).transpose(0, 1), targets, torch.tensor([20, 16]), torch.tensor([3, 3]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])


class TestCurriculum(unittest.TestCase):
    def test_starts_with_shortest_third_and_grows_to_all(self):
        from main import curriculum_indices

        lengths = [9, 1, 5, 3, 7, 2]
        ds = SimpleNamespace(samples=[{"label": torch.zeros(n)} for n in lengths])
        self.assertEqual(curriculum_indices(ds, 1, 3), [1, 5])          # the two shortest of six
        self.assertEqual(len(curriculum_indices(ds, 3, 3)), 5)
        self.assertIsNone(curriculum_indices(ds, 4, 3))                 # after the curriculum: all clips
        self.assertIsNone(curriculum_indices(ds, 1, 0))                 # off


if __name__ == "__main__":
    unittest.main()

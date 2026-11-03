"""End-to-end CPU smoke test: dataset -> train -> eval -> infer.

Skipped by default. Run with:
    RUN_SLOW=1 python -m unittest tests.test_smoke_e2e -v
"""

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import torch

from Data.synthetic import create_synthetic_dataset
from config import get_hindi_vocab
from main import run_evaluation, run_inference, train


@unittest.skipUnless(os.environ.get("RUN_SLOW") == "1", "set RUN_SLOW=1 to run the E2E smoke test")
class TestEndToEndSmoke(unittest.TestCase):
    def test_train_eval_infer(self):
        tmp = tempfile.mkdtemp(prefix="vallr_e2e_")
        try:
            create_synthetic_dataset(
                tmp,
                speakers=2,
                clips=1,
                seconds=0.6,
                fps=25,
                size=(48, 48),
                texts=("नमस्ते",),
            )
            checkpoint = os.path.join(tmp, "tiny.pth")
            video_path = os.path.join(tmp, "spk01", "spk01_001.mp4")
            args = SimpleNamespace(
                data_dir=tmp,
                videos_root=None,
                val_dir=None,
                save_model_path=checkpoint,
                sarvam_adapter_path=None,
                infer_video_path=video_path,
                val_split=0.5,
                sample_size=1.0,
                num_frames=0,
                max_frames=16,
                frame_stride=1,
                crop_mode="none",
                preprocessed=False,
                crop_face=False,
                allow_unknown_speakers=False,
                max_invalid_sample_pct=100.0,
                strict_ctc=False,
                temporal_upsample_factor=2,
                vit_embed_dim=32,
                vit_depth=1,
                vit_heads=4,
                spatial_pool_tokens="none",
                batch_size=1,
                eval_batch_size=1,
                num_workers=0,
                epochs=1,
                warmup_steps=1,
                device="cpu",
                language="hindi",
                mode="train",
            )
            vocab = get_hindi_vocab()
            device = torch.device("cpu")

            train(args, device, vocab)
            self.assertTrue(os.path.exists(checkpoint))

            run_evaluation(args, device, vocab)
            result = run_inference(
                video_path=video_path,
                stage1_model_path=checkpoint,
                sarvam_adapter_path=None,
                device=device,
                vocab=vocab,
                temporal_upsample_factor=2,
                max_frames=16,
                crop_mode="none",
                args=args,
            )
            self.assertIsNotNone(result)
            self.assertIn("phonemes", result)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

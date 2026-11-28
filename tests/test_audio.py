"""Audio features (scripts/extract_audio.py), the audio-only model and audio-guided lip training."""

import os
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from Models.ML_VALLR import VALLR, stage1_kwargs_from_state_dict
from scripts.extract_audio import MELS_PER_FRAME, frame_features, log_mel, process_clip


class TestFeatures(unittest.TestCase):
    def test_log_mel_is_100_frames_per_second(self):
        sr = 16000
        mel = log_mel(np.sin(2 * np.pi * 440 * np.arange(sr) / sr))  # 1 s tone
        self.assertEqual(mel.shape, (100, 80))
        loud = mel.mean(axis=0).argmax()
        self.assertLess(loud, 20)  # 440 Hz lands in a low band

    def test_frame_features_line_up_with_video_frames(self):
        mel = np.arange(30 * 80, dtype=np.float32).reshape(30, 80)
        out = frame_features(mel, 8)  # 8 frames need 32 mel frames: two are padded
        self.assertEqual(out.shape, (8, MELS_PER_FRAME, 80))
        self.assertTrue(np.array_equal(out[2, 0], mel[8]))  # video frame 2 = mel frames 8..11

    @unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
    def test_process_clip_writes_features_next_to_the_crop(self):
        tmp = tempfile.mkdtemp()
        try:
            src, dst = os.path.join(tmp, "raw", "spk01"), os.path.join(tmp, "crop")
            os.makedirs(src)
            clip = os.path.join(src, "spk01_001.mp4")
            subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=25:duration=1",
                            "-f", "lavfi", "-i", "sine=frequency=300:duration=1", "-shortest", clip], check=True)
            self.assertEqual(process_clip(clip, os.path.join(tmp, "raw"), dst, 96, False), (clip, None))
            feats = np.load(os.path.join(dst, "spk01", "spk01_001.audio.npy"))
            self.assertEqual(feats.shape[1:], (4, 80))
            self.assertIn(feats.shape[0], (25, 26))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestAudioModel(unittest.TestCase):
    def test_audio_model_ctc_length_and_checkpoint_detection(self):
        torch.manual_seed(0)
        model = VALLR(num_classes=12, frontend="audio", backend_dim=64, backend_layers=1).eval()
        for frames in (5, 8):
            with torch.no_grad():
                logits, _ = model(torch.randn(1, frames, 320), torch.tensor([frames]))
            self.assertEqual(logits.shape[1], -(-frames // 2) * 4)
        kw = stage1_kwargs_from_state_dict(model.state_dict())
        self.assertEqual((kw["frontend"], kw["audio_head"]), ("audio", False))
        VALLR(num_classes=12, **kw).load_state_dict(model.state_dict())

    def test_resnet_audio_head_predicts_every_frame(self):
        model = VALLR(num_classes=12, frontend="resnet", backend_dim=64, backend_layers=1, audio_head=True).eval()
        with torch.no_grad():
            model(torch.rand(2, 7, 3, 64, 64), torch.tensor([7, 5]))
        self.assertEqual(tuple(model.last_audio_pred.shape), (2, 8, 320))  # 7 frames padded to the tubelet (8)
        kw = stage1_kwargs_from_state_dict(model.state_dict())
        self.assertEqual((kw["frontend"], kw["audio_head"]), ("resnet", True))

    def test_audio_prediction_loss_ignores_padding(self):
        from main import _audio_prediction_loss

        model = SimpleNamespace(last_audio_pred=torch.zeros(2, 4, 320))
        audio = torch.zeros(2, 4, 320)
        audio[0, :2] = 1.0  # sample 0 has 2 real frames, both off by 1
        audio[1, 3] = 9.0   # a padded frame of sample 1: must be ignored
        batch = SimpleNamespace(audio=audio)
        loss = _audio_prediction_loss(model, batch, torch.tensor([2, 3]), torch.device("cpu"))
        self.assertAlmostEqual(loss.item(), 2 / 5)  # 2 wrong frames out of 5 real ones


class TestDistillation(unittest.TestCase):
    def test_distill_loss_is_zero_for_a_perfect_copy_and_ignores_padding(self):
        from main import _distill_loss

        teacher = torch.randn(2, 6, 12)
        self.assertAlmostEqual(_distill_loss(teacher, teacher, torch.tensor([6, 4]), torch.device("cpu")).item(), 0.0,
                               places=5)
        student = teacher.clone()
        student[1, 4:] = torch.randn(2, 12) * 5  # only padded steps of clip 1 differ
        self.assertAlmostEqual(_distill_loss(student, teacher, torch.tensor([6, 4]), torch.device("cpu")).item(), 0.0,
                               places=5)
        student[0, 0] = torch.randn(12) * 5
        self.assertGreater(_distill_loss(student, teacher, torch.tensor([6, 4]), torch.device("cpu")).item(), 0.0)

    def test_teacher_must_be_an_audio_model(self):
        from main import load_teacher

        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "lip.pth")
            torch.save(VALLR(num_classes=12, frontend="resnet", backend_dim=64, backend_layers=1).state_dict(), path)
            args = SimpleNamespace(temporal_upsample_factor=4, vit_embed_dim=32, vit_depth=1, vit_heads=4,
                                   spatial_pool_tokens="none", frontend="resnet", backend_dim=64, backend_layers=1,
                                   audio_weight=0.0)
            with self.assertRaises(ValueError):
                load_teacher(path, torch.device("cpu"), 12, args)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(os.environ.get("RUN_SLOW") == "1", "set RUN_SLOW=1 to run")
class TestAudioTrainingE2E(unittest.TestCase):
    def test_audio_only_and_audio_guided_training(self):
        from config import get_hindi_vocab
        from Data.synthetic import create_synthetic_dataset
        from main import run_evaluation, train

        tmp = tempfile.mkdtemp()
        try:
            create_synthetic_dataset(tmp, speakers=2, clips=2, seconds=0.8, size=(48, 48))
            rng = np.random.RandomState(0)
            for root, _, files in os.walk(tmp):
                for f in files:
                    if f.endswith(".mp4"):
                        np.save(os.path.join(root, f[:-4] + ".audio.npy"),
                                rng.randn(20, 4, 80).astype(np.float16))
            base = dict(data_dir=tmp, videos_root=None, val_dir=None, sarvam_adapter_path=None, val_split=0.5,
                        sample_size=1.0, num_frames=0, max_frames=24, frame_stride=1, crop_mode="none",
                        preprocessed=False, crop_face=False, allow_unknown_speakers=False,
                        max_invalid_sample_pct=100.0, strict_ctc=False, temporal_upsample_factor=4,
                        vit_embed_dim=32, vit_depth=1, vit_heads=4, spatial_pool_tokens="none", batch_size=2,
                        eval_batch_size=2, num_workers=0, epochs=1, warmup_steps=1, lr_patience=3, min_lr=1e-5,
                        lr=3e-4, resume=False, init_from=None, backend_dim=64, backend_layers=1,
                        curriculum_epochs=0, device="cpu", language="hindi", mode="train")
            vocab, dev = get_hindi_vocab(), torch.device("cpu")
            audio_args = SimpleNamespace(**base, frontend="audio", audio_weight=0.0,
                                         save_model_path=os.path.join(tmp, "audio.pth"))
            train(audio_args, dev, vocab)
            run_evaluation(audio_args, dev, vocab)
            lip_args = SimpleNamespace(**base, frontend="resnet", audio_weight=1.0,
                                       save_model_path=os.path.join(tmp, "lip.pth"))
            train(lip_args, dev, vocab)
            state = torch.load(os.path.join(tmp, "lip.pth"), weights_only=True)
            self.assertIn("audio_head.weight", state)
            student = SimpleNamespace(**base, frontend="resnet", audio_weight=0.0,
                                      teacher_path=os.path.join(tmp, "audio.pth"), distill_weight=1.0,
                                      save_model_path=os.path.join(tmp, "student.pth"))
            train(student, dev, vocab)
            self.assertNotIn("audio_head.weight", torch.load(os.path.join(tmp, "student.pth"), weights_only=True))

            # Starting a lip model from the audio model copies everything but the front end.
            from main import ML_VALLR, _model_kwargs  # noqa: F401
            warm = SimpleNamespace(**base, frontend="resnet", audio_weight=0.0, teacher_path=None,
                                   save_model_path=os.path.join(tmp, "warm.pth"))
            warm.init_from, warm.epochs, warm.lr = os.path.join(tmp, "audio.pth"), 1, 1e-12
            train(warm, dev, vocab)
            a = torch.load(os.path.join(tmp, "audio.pth"), weights_only=True)
            w = torch.load(os.path.join(tmp, "warm_last.pt"), weights_only=False)["model"]
            self.assertTrue(torch.allclose(a["ctc_head.weight"], w["ctc_head.weight"], atol=1e-6))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

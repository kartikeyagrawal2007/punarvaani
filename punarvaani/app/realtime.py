"""Live demo: webcam -> MediaPipe landmarks -> online crop -> end-pointing -> Path A -> speaker.

The week-14 plan is phone camera -> laptop GPU -> audio back. This script is
the laptop half running on a local webcam; the phone half can stream frames
over a WebSocket into the same :class:`LiveSession.push_frame`.

    python -m punarvaani.app.realtime --run outputs/hindi_adapt --phrases configs/phrases_hi.txt \\
        --face-model face_landmarker.task
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from punarvaani.app.streaming import OnlineLipCropper, UtteranceStreamer
from punarvaani.models.path_a.phrase_mode import PhraseBank
from punarvaani.models.path_a.pipeline import PathAPipeline
from punarvaani.models.path_a.tts import CachedTTS, ToneTTS
from punarvaani.text.vocab import CharVocab
from punarvaani.training.train_vsr import build_vsr
from punarvaani.utils import get_logger, load_config

log = get_logger(__name__)


class LiveSession:
    def __init__(self, pipeline: PathAPipeline, landmarker, play=None):
        self.pipeline, self.landmarker = pipeline, landmarker
        self.cropper = OnlineLipCropper()
        self.play = play or (lambda wav, sr: None)
        self.streamer = UtteranceStreamer(self._on_utterance)

    def _on_utterance(self, video: torch.Tensor):
        t0 = time.perf_counter()
        result = self.pipeline.recognize(video)
        log.info(
            "[%s] %s (%.2f) in %.0f ms",
            result.mode,
            result.text,
            result.confidence,
            (time.perf_counter() - t0) * 1e3,
        )
        if result.audio is not None:
            self.play(result.audio, result.sample_rate)
        return result

    def push_frame(self, frame_rgb: np.ndarray):
        crop = self.cropper(frame_rgb, self.landmarker(frame_rgb))
        return self.streamer.push(crop) if crop is not None else None


def load_pipeline(run_dir: str | Path, phrases: str | None = None, device: str = "cpu") -> PathAPipeline:
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    vocab = CharVocab.load(run_dir / "vocab.json")
    model = build_vsr(cfg, vocab)
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    bank = PhraseBank.load(phrases, vocab) if phrases else None
    tts = CachedTTS(ToneTTS())
    return PathAPipeline(model.to(device), vocab, bank, tts=tts)


def main(argv=None) -> None:
    import cv2

    from punarvaani.preprocessing.lip_crop import MediaPipeLandmarker

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training output dir with config.yaml, vocab.json, best.pt")
    ap.add_argument("--phrases")
    ap.add_argument("--face-model", required=True)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args(argv)

    try:
        import sounddevice as sd

        play = lambda wav, sr: sd.play(wav.numpy(), sr)  # noqa: E731
    except ImportError:
        play = None
    session = LiveSession(load_pipeline(args.run, args.phrases), MediaPipeLandmarker(args.face_model), play)
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FPS, 25)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        session.push_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cv2.imshow("punarvaani", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()


if __name__ == "__main__":
    main()

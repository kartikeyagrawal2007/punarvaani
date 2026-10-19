import numpy as np
import torch

from punarvaani.app.streaming import ChunkedUnitStreamer, LipActivityDetector, UtteranceStreamer
from punarvaani.models.encoder.avhubert import EncoderConfig, VisualSpeechEncoder
from punarvaani.models.path_b.model import LipToUnitModel


def _crops(pattern, seed=0):
    """Static noisy mouth when 0, strongly moving mouth when 1."""
    rng = np.random.default_rng(seed)
    base = rng.integers(100, 110, (44, 44)).astype(np.float32)
    out = []
    for k, moving in enumerate(pattern):
        img = base + rng.normal(0, 0.5, base.shape)
        if moving:
            h = 6 + 8 * abs(np.sin(k * 0.9))
            img[22 - int(h) : 22 + int(h), 12:32] = 20
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    return out


def test_activity_detector_finds_one_utterance_despite_short_closures():
    pattern = [0] * 40 + [1] * 10 + [0] * 3 + [1] * 15 + [0] * 40  # 3-frame closure (like "म") inside
    det = LipActivityDetector()
    events = [e for e in (det(c) for c in _crops(pattern)) if e]
    assert [e.kind for e in events] == ["start", "end"]
    assert 38 <= events[0].frame <= 44 and 64 <= events[1].frame <= 72


def test_utterance_streamer_emits_complete_utterance():
    got = []
    streamer = UtteranceStreamer(lambda video: got.append(video.shape) or "ok")
    for c in _crops([0] * 40 + [1] * 20 + [0] * 30):
        streamer.push(c)
    assert len(got) == 1 and 20 <= got[0][0] <= 40 and got[0][1:] == (44, 44)


def test_chunked_streamer_covers_every_frame_once():
    torch.manual_seed(0)
    model = LipToUnitModel(
        VisualSpeechEncoder(EncoderConfig.preset("tiny")), num_units=10, hidden=16, predict_mel=False
    )
    streamer = ChunkedUnitStreamer(model, chunk=5, left=10, right=2)
    units = []
    for _ in range(23):
        out = streamer.push(torch.randn(44, 44))
        if out:
            units.append(out[0])
    units.append(streamer.flush()[0])
    assert sum(u.numel() for u in units) == 2 * 23
    assert streamer.latency_frames == 7

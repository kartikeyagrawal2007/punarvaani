import numpy as np
import torch

from punarvaani.data_engine.pipeline import MiningPipeline, split_by_group
from punarvaani.data_engine.segment import Word, segment_words
from punarvaani.data_engine.sync import SyncNet, av_offset, contrastive_sync_loss
from punarvaani.data_engine.tracking import IoUTracker, shot_boundaries, usable_tracks
from punarvaani.datasets.av_dataset import AVManifestDataset, read_manifest
from punarvaani.preprocessing.lip_crop import CANONICAL_STABLE, LIPS_INNER, LIPS_OUTER, STABLE_POINTS
from punarvaani.text.vocab import CharVocab


def _words(spec):
    return [Word(t, s, e) for t, s, e in spec]


def test_segment_words_breaks_at_pauses_and_respects_max():
    words = _words([("मुझे", 0.0, 0.4), ("पानी", 0.45, 0.9), ("चाहिए", 0.95, 1.5),  # pause
                    ("दर्द", 2.6, 3.0), ("है", 3.05, 3.6)])  # fmt: skip
    clips = segment_words(words, min_s=0.5)
    assert [c.text for c in clips] == ["मुझे पानी चाहिए", "दर्द है"]
    long = _words([(f"w{i}", i * 0.5, i * 0.5 + 0.45) for i in range(60)])
    assert all(c.duration <= 12.0 + 0.3 for c in segment_words(long))


def test_av_offset_finds_shift_and_confidence():
    torch.manual_seed(0)
    v = torch.randn(80, 16)
    a = torch.roll(v, 3, dims=0) + 0.05 * torch.randn(80, 16)  # audio lags video by 3 frames
    res = av_offset(v, a, max_offset=10)
    assert res.offset == 3 and res.confidence > 1.0
    noise = av_offset(v, torch.randn(80, 16), max_offset=10)
    assert noise.confidence < res.confidence


def test_syncnet_shapes_and_loss():
    net = SyncNet(dim=32, n_mels=40, width=4)
    v, a = net.clip_embeddings(torch.randn(12, 48, 48), torch.randn(40, 48))
    assert v.shape == a.shape == (12, 32)
    assert torch.isfinite(contrastive_sync_loss(v, a))


def test_shots_and_tracking():
    frames = np.zeros((20, 32, 32, 3), dtype=np.uint8)
    frames[10:] = 200  # a cut at frame 10
    assert shot_boundaries(frames) == [0, 10]
    dets = [[(10 + t * 0.5, 10, 50 + t * 0.5, 60)] for t in range(20)]
    tracks = IoUTracker()(dets, shot_starts=[0, 10])
    assert [tr.length for tr in tracks] == [10, 10]
    assert len(usable_tracks(tracks, frame_height=100, min_frames=8)) == 2


def test_split_is_speaker_disjoint_and_deterministic():
    rows = [{"id": i, "speaker": f"spk{i % 50}"} for i in range(1000)]
    a, b = split_by_group(rows, val=0.1, test=0.1), split_by_group(rows, val=0.1, test=0.1)
    assert a == b
    spk = {k: {r["speaker"] for r in v} for k, v in a.items()}
    assert not (spk["train"] & spk["test"]) and not (spk["train"] & spk["val"])


def _fake_landmarker(frames):
    lm = np.zeros((478, 2))
    lm[list(STABLE_POINTS)] = CANONICAL_STABLE * 0.5
    ang = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    lm[list(LIPS_OUTER)] = np.array([64, 95]) + np.stack([15 * np.cos(ang), 6 * np.sin(ang)], 1)
    lm[list(LIPS_INNER)] = np.array([64, 95]) + np.stack([10 * np.cos(ang), 3 * np.sin(ang)], 1)
    return np.repeat(lm[None], len(frames), 0), np.ones(len(frames), dtype=bool)


def test_mining_pipeline_end_to_end_with_fakes(tmp_path):
    frames = np.random.default_rng(0).integers(
        90, 110, size=(100, 128, 128, 3), dtype=np.uint8
    )  # 4 s, one shot
    audio = torch.zeros(4 * 16000)
    words = _words([("मुझे", 0.2, 0.6), ("पानी", 0.65, 1.1), ("चाहिए", 1.15, 1.7),
                    ("please", 2.5, 2.9), ("call", 2.95, 3.4)])  # fmt: skip
    pipe = MiningPipeline(tmp_path, _fake_landmarker, lambda wav: words)
    rows, report = pipe.process_video("vid1", frames, audio, speaker="anchor1", source="test")
    assert report.kept == 1 and report.dropped["code_mixed_or_non_hindi"] == 1
    from punarvaani.datasets.av_dataset import write_manifest

    write_manifest(rows, tmp_path / "m.jsonl")
    ds = AVManifestDataset(read_manifest(tmp_path / "m.jsonl"), CharVocab())
    item = ds[0]
    assert item["text"] == "मुझे पानी चाहिए" and item["video"].shape[1:] == (88, 88)

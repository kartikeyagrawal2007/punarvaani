import math

import torch

from punarvaani.voice_rebuild.denoise import estimate_bandwidth, quality_report, spectral_wiener
from punarvaani.voice_rebuild.rebuild import VoiceRebuilder, agglomerative_cosine, speech_regions
from punarvaani.voice_rebuild.speaker import ECAPATDNN, AAMSoftmax

SR = 16000


def _voice(f0, seconds, seed=0):
    """A crude 'voice': harmonic stack at f0 with syllable-like amplitude modulation."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(int(seconds * SR)) / SR
    wav = sum(torch.sin(2 * math.pi * f0 * k * t + torch.rand(1, generator=g) * 6) / k for k in range(1, 6))
    return 0.2 * wav * (0.6 + 0.4 * torch.sin(2 * math.pi * 4 * t)).clamp(min=0)


def test_wiener_improves_snr():
    clean = _voice(150, 2.0)
    noisy = torch.cat([torch.zeros(SR), clean]) + 0.02 * torch.randn(3 * SR)
    before, after = quality_report(noisy), quality_report(spectral_wiener(noisy))
    assert after.snr_db > before.snr_db + 3


def test_bandwidth_detects_narrowband():
    t = torch.arange(SR) / SR
    wide = torch.randn(SR) * 0.1
    narrow = sum(torch.sin(2 * math.pi * f * t) for f in (300, 900, 2000, 3500)) * 0.1
    assert estimate_bandwidth(wide) > 7000 and estimate_bandwidth(narrow) < 5000


def test_speech_regions_merge_short_gaps():
    wav = torch.cat(
        [_voice(150, 1.0), torch.zeros(int(0.1 * SR)), _voice(150, 1.0), torch.zeros(SR), _voice(150, 1.0)]
    )
    regions = speech_regions(wav)
    assert len(regions) == 2


def test_ecapa_shapes_and_aam_loss():
    model = ECAPATDNN.tiny()
    emb = model(torch.randn(3, SR))
    assert emb.shape == (3, 32)
    loss = AAMSoftmax(32, 5)(emb, torch.tensor([0, 1, 2]))
    loss.backward()
    assert torch.isfinite(loss) and model.fc.weight.grad is not None


def test_clustering_separates_two_groups():
    torch.manual_seed(0)
    a, b = torch.randn(16), torch.randn(16)
    x = torch.stack(
        [a + 0.05 * torch.randn(16) for _ in range(5)] + [b + 0.05 * torch.randn(16) for _ in range(3)]
    )
    labels = agglomerative_cosine(x, 0.3)
    assert labels[:5] == [0] * 5 and labels[5:] == [1] * 3


def _pitch_embedder(wav):
    """Toy speaker embedder: a 2-D vector from the dominant frequency (stands in for ECAPA)."""
    spec = torch.fft.rfft(wav).abs()
    f = float(spec[50:].argmax() + 50) * SR / wav.numel()
    return torch.tensor([math.cos(f / 100), math.sin(f / 100)])


def test_rebuild_keeps_the_patient_not_the_relatives(tmp_path):
    patient = [_voice(120, 3.0, s) for s in range(3)]
    relative = _voice(260, 3.0, 9)
    wavs = [
        ("note1", torch.cat([patient[0], torch.zeros(SR), relative])),
        ("note2", patient[1]),
        ("video", patient[2]),
    ]
    rebuilder = VoiceRebuilder(_pitch_embedder, denoise_backend="none", cluster_threshold=0.2, min_quality=-10,
                               transcriber=lambda w: "नमस्ते")  # fmt: skip
    profile = rebuilder.build(wavs, tmp_path)
    assert profile.report["clusters"] == 2
    kept = [s for s in profile.segments if s["cluster"] == profile.report["target_cluster"]]
    assert all(s["source"] != "note1" or s["start_s"] < 3.0 for s in kept)  # relative's part excluded
    assert profile.reference_text == "नमस्ते" and (tmp_path / "speaker_embedding.pt").exists()

    # with an enrolment clip of the relative, the relative's cluster is chosen instead
    other = rebuilder.build(wavs, tmp_path / "other", enrollment=_voice(260, 2.0, 4))
    assert other.report["target_cluster"] != profile.report["target_cluster"]

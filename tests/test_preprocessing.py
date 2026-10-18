import math

import numpy as np
import torch

from punarvaani.preprocessing.audio import LogMel, energy_vad, mel_filterbank, resample, stack_frames
from punarvaani.preprocessing.lip_crop import (
    CANONICAL_STABLE,
    LIPS_INNER,
    LIPS_OUTER,
    STABLE_POINTS,
    LipCropper,
    apply_transform,
    mouth_opening,
    similarity_transform,
    smooth_landmarks,
)


def _random_similarity(rng):
    angle, scale = rng.uniform(-0.3, 0.3), rng.uniform(0.8, 1.6)
    c, s = math.cos(angle), math.sin(angle)
    m = np.zeros((2, 3))
    m[:, :2] = scale * np.array([[c, -s], [s, c]])
    m[:, 2] = rng.uniform(20, 120, size=2)
    return m


def test_similarity_transform_recovers_known_transform():
    rng = np.random.default_rng(0)
    m = _random_similarity(rng)
    src = rng.normal(size=(6, 2)) * 50
    est = similarity_transform(src, apply_transform(src, m))
    assert np.allclose(est, m, atol=1e-8)


def test_smoothing_reduces_jitter_and_keeps_length():
    rng = np.random.default_rng(0)
    x = np.zeros((50, 3, 2)) + rng.normal(scale=2.0, size=(50, 3, 2))
    y = smooth_landmarks(x, window=12)
    assert y.shape == x.shape
    assert y.std() < x.std() / 2


def _synthetic_face_video(rng, t=20, size=200):
    """Frames with a bright square painted at the mouth; the head moves every frame."""
    canonical = np.zeros((478, 2))
    canonical[list(STABLE_POINTS)] = CANONICAL_STABLE
    mouth_centre = np.array([128.0, 190.0])
    ang = np.linspace(0, 2 * np.pi, len(LIPS_OUTER), endpoint=False)
    canonical[list(LIPS_OUTER)] = mouth_centre + np.stack([30 * np.cos(ang), 12 * np.sin(ang)], 1)
    canonical[list(LIPS_INNER)] = mouth_centre + np.stack([20 * np.cos(ang), 5 * np.sin(ang)], 1)
    frames = np.zeros((t, size, size, 3), dtype=np.uint8)
    landmarks = np.zeros((t, 478, 2))
    base = _random_similarity(rng) * np.array([[0.5], [0.5]])  # shrink so the face fits
    base[:, 2] = [20, 0]
    for i in range(t):
        m = base.copy()
        m[:, 2] += [i * 0.8, i * 0.5]  # slow head drift
        landmarks[i] = apply_transform(canonical, m)
        cx, cy = apply_transform(mouth_centre[None], m)[0].astype(int)
        frames[i, cy - 4 : cy + 5, cx - 4 : cx + 5] = 255
    return frames, landmarks


def test_lip_cropper_centres_the_mouth_despite_head_motion():
    frames, landmarks = _synthetic_face_video(np.random.default_rng(1))
    result = LipCropper(crop_size=96, smooth_window=1)(frames, landmarks)
    assert result.crops.shape == (20, 96, 96) and result.crops.dtype == np.uint8
    for crop in result.crops:
        ys, xs = np.nonzero(crop > 128)
        assert abs(ys.mean() - 47.5) < 3 and abs(xs.mean() - 47.5) < 3
    opening = mouth_opening(result.mouth_landmarks)
    assert np.allclose(opening, 10 / 60, atol=0.02)


def test_lip_cropper_interpolates_lost_frames():
    frames, landmarks = _synthetic_face_video(np.random.default_rng(2))
    valid = np.ones(20, dtype=bool)
    valid[7] = False
    landmarks[7] = 0  # detector failed
    result = LipCropper(smooth_window=1)(frames, landmarks, valid)
    assert abs(result.missing_ratio - 0.05) < 1e-9
    assert np.isfinite(result.mouth_landmarks).all()


def test_mel_filterbank_and_frame_rates():
    fb = mel_filterbank(80, 640, 16000)
    assert fb.shape == (80, 321) and (fb >= 0).all()
    wav = torch.randn(16000)
    mel = LogMel(center=False)(wav)
    assert mel.shape == (80, 100)  # 100 Hz
    assert stack_frames(mel.T).shape == (25, 320)  # 25 Hz, 4 mel frames per video frame


def test_resample_preserves_tone_frequency():
    t = torch.arange(44100) / 44100
    wav = torch.sin(2 * math.pi * 440 * t)
    out = resample(wav, 44100, 16000)
    assert abs(out.numel() - 16000) <= 1
    spectrum = torch.fft.rfft(out).abs()
    assert abs(int(spectrum.argmax()) - 440) <= 2


def test_energy_vad_finds_speech_region():
    wav = torch.zeros(32000)
    wav[8000:24000] = 0.3 * torch.randn(16000)
    vad = energy_vad(wav)
    assert vad[60:140].float().mean() > 0.9
    assert vad[:40].float().mean() < 0.1

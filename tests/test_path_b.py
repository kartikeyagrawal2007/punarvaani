import math

import torch

from punarvaani.datasets.av_dataset import collate_av
from punarvaani.datasets.synthetic import SyntheticLipDataset
from punarvaani.models.encoder.avhubert import EncoderConfig, VisualSpeechEncoder
from punarvaani.models.path_b.model import LipToUnitModel
from punarvaani.models.path_b.units import KMeans, dedup_units
from punarvaani.models.path_b.vocoder import Discriminators, UnitHiFiGAN, VocoderLoss
from punarvaani.preprocessing.pitch import pitch_targets, yin


def test_kmeans_recovers_clusters():
    torch.manual_seed(0)
    centres = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    x = torch.cat([c + 0.3 * torch.randn(200, 2) for c in centres])
    km = KMeans(3).fit(x)
    labels = km.predict(x)
    for i in range(3):
        assert labels[i * 200 : (i + 1) * 200].unique().numel() == 1
    assert labels.unique().numel() == 3


def test_dedup_units():
    u, d = dedup_units(torch.tensor([5, 5, 5, 9, 9, 2]))
    assert u.tolist() == [5, 9, 2] and d.tolist() == [3, 2, 1]


def test_yin_tracks_a_tone_and_flags_silence():
    sr = 16000
    t = torch.arange(sr) / sr
    wav = torch.cat([0.5 * torch.sin(2 * math.pi * 180 * t), torch.zeros(sr // 2)])
    f0, voiced = yin(wav)
    assert f0.shape[0] == wav.numel() // 320
    assert abs(f0[5:45].median().item() - 180) < 2
    assert voiced[5:45].all() and not voiced[-10:].any()
    tgt = pitch_targets(f0, voiced)
    assert tgt.shape == (f0.shape[0], 2) and tgt[-1, 1] == 0


def test_lip_to_unit_losses_and_rates():
    ds = SyntheticLipDataset(3, size=32)
    batch = collate_av([ds[i] for i in range(3)], ds.vocab.pad_id)
    enc = VisualSpeechEncoder(EncoderConfig.preset("tiny"))
    model = LipToUnitModel(
        enc, num_units=len(ds.vocab), hidden=32, predict_pitch=True, ctc_vocab_size=len(ds.vocab)
    )
    t = batch["video"].shape[1]
    pitch = torch.zeros(3, 2 * t, 2)
    out = model(
        batch["video"],
        batch["lengths"],
        batch["units"],
        batch["mel"],
        pitch,
        batch["tokens"],
        batch["token_lengths"],
    )
    assert out.unit_logits.shape == (3, 2 * t, len(ds.vocab)) and out.mel.shape == (3, 4 * t, 80)
    assert torch.isfinite(out.loss) and {"unit_ce", "mel_l1", "pitch", "ctc", "unit_acc"} <= set(out.logs)
    out.loss.backward()
    assert model.predict_units(batch["video"][0], int(batch["lengths"][0])).shape == (
        2 * int(batch["lengths"][0]),
    )


def test_vocoder_shapes_and_gan_step():
    gen = UnitHiFiGAN.tiny(num_units=20, speaker_dim=8, use_pitch=True)
    disc = Discriminators.tiny()
    units = torch.randint(0, 20, (2, 10))
    wav = gen(units, torch.randn(2, 8), torch.zeros(2, 10, 2))
    assert gen.hop == 320 and wav.shape == (2, 1, 3200)
    real = 0.1 * torch.randn(2, 1, 3200)
    losses = VocoderLoss()
    d_loss = losses.discriminator_step(disc, real, wav)
    g_loss, logs = losses.generator_step(disc, real, wav)
    assert torch.isfinite(d_loss) and torch.isfinite(g_loss) and logs["mel_l1"] > 0
    g_loss.backward()
    assert gen.unit_embed.weight.grad is not None

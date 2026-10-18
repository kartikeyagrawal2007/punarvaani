"""Unit HiFi-GAN: speech units (+ speaker embedding, + pitch) -> 16 kHz waveform.

HiFi-GAN (Kong et al. 2020) is a GAN vocoder: a generator of transposed convs
upsamples a frame-rate input to samples; "multi-receptive-field" residual
blocks with different kernels/dilations model periodic and noisy parts of
speech. Two discriminator families judge realism:

* **MPD** (multi-period): reshapes the waveform into 2-D with period p ∈
  {2,3,5,7,11} so each sub-discriminator sees periodic structure (pitch
  harmonics) at a different period;
* **MSD** (multi-scale): looks at raw, 2x and 4x downsampled audio for
  structure at different time scales.

Generator loss = adversarial (LSGAN) + 2·feature matching + 45·mel L1 - the
mel term gives stable early training, the GAN terms add the crispness that
L1 alone blurs away.

Speech-resynthesis (Polyak et al. 2021) made the input *discrete units +
speaker embedding (+ F0)*. The speaker embedding is why Path B can speak in the
patient's voice: train the vocoder on many speakers, then fine-tune on the
patient's cleaned voice notes - **audio only**, no video needed, because units
for those notes come straight from mHuBERT.

Upsampling 5·4·4·2·2 = 320 samples per unit = 20 ms at 16 kHz.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from punarvaani.preprocessing.audio import LogMel

LRELU = 0.1


def _pad(kernel: int, dilation: int = 1) -> int:
    return (kernel * dilation - dilation) // 2


class ResBlock(nn.Module):
    def __init__(self, ch: int, kernel: int, dilations: Sequence[int]):
        super().__init__()
        self.convs1 = nn.ModuleList(
            weight_norm(nn.Conv1d(ch, ch, kernel, 1, _pad(kernel, d), dilation=d)) for d in dilations
        )
        self.convs2 = nn.ModuleList(
            weight_norm(nn.Conv1d(ch, ch, kernel, 1, _pad(kernel))) for _ in dilations
        )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            x = x + c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU)), LRELU))
        return x


class UnitHiFiGAN(nn.Module):
    def __init__(
        self,
        num_units: int = 1000,
        unit_dim: int = 128,
        speaker_dim: int = 192,
        speaker_proj: int = 128,
        use_pitch: bool = False,
        upsample_rates: Sequence[int] = (5, 4, 4, 2, 2),
        upsample_kernels: Sequence[int] = (11, 8, 8, 4, 4),
        channels: int = 512,
        resblock_kernels: Sequence[int] = (3, 7, 11),
        resblock_dilations: Sequence[Sequence[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        self.hop = 1
        for r in upsample_rates:
            self.hop *= r
        self.unit_embed = nn.Embedding(num_units, unit_dim)
        self.speaker_proj = nn.Linear(speaker_dim, speaker_proj) if speaker_dim else None
        self.use_pitch = use_pitch
        in_ch = unit_dim + (speaker_proj if speaker_dim else 0) + (2 if use_pitch else 0)
        self.conv_pre = weight_norm(nn.Conv1d(in_ch, channels, 7, 1, 3))
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        ch = channels
        for r, k in zip(upsample_rates, upsample_kernels):
            self.ups.append(weight_norm(nn.ConvTranspose1d(ch, ch // 2, k, r, padding=(k - r) // 2)))
            ch //= 2
            for rk, rd in zip(resblock_kernels, resblock_dilations):
                self.resblocks.append(ResBlock(ch, rk, rd))
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, 3))
        self.num_kernels = len(resblock_kernels)

    def forward(
        self, units: torch.Tensor, speaker: torch.Tensor | None = None, pitch: torch.Tensor | None = None
    ):
        """``units`` (B, T) long; ``speaker`` (B, speaker_dim); ``pitch`` (B, T, 2). -> (B, 1, T * hop)."""
        x = self.unit_embed(units.clamp(min=0)).transpose(1, 2)
        parts = [x]
        if self.speaker_proj is not None:
            if speaker is None:
                raise ValueError("this vocoder needs a speaker embedding")
            parts.append(self.speaker_proj(speaker).unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        if self.use_pitch:
            parts.append(
                (
                    pitch if pitch is not None else torch.zeros(x.shape[0], x.shape[-1], 2, device=x.device)
                ).transpose(1, 2)
            )
        x = self.conv_pre(torch.cat(parts, 1))
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, LRELU))
            x = (
                sum(self.resblocks[i * self.num_kernels + j](x) for j in range(self.num_kernels))
                / self.num_kernels
            )
        return torch.tanh(self.conv_post(F.leaky_relu(x)))

    @classmethod
    def tiny(cls, num_units: int, speaker_dim: int = 0, use_pitch: bool = False) -> UnitHiFiGAN:
        return cls(num_units, unit_dim=32, speaker_dim=speaker_dim, speaker_proj=16, use_pitch=use_pitch, channels=64,
                   resblock_kernels=(3,), resblock_dilations=((1, 3),))  # fmt: skip


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, width: int = 32):
        super().__init__()
        self.period = period
        chs = [1, width, width * 4, width * 16, width * 32, width * 32]
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(chs[i], chs[i + 1], (5, 1), (3 if i < 4 else 1, 1), (2, 0)))
            for i in range(5)
        )
        self.conv_post = weight_norm(nn.Conv2d(chs[-1], 1, (3, 1), 1, (1, 0)))

    def forward(self, x):
        b, c, t = x.shape
        if t % self.period:
            x = F.pad(x, (0, self.period - t % self.period), "reflect")
        x = x.view(b, c, -1, self.period)
        fmaps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU)
            fmaps.append(x)
        x = self.conv_post(x)
        fmaps.append(x)
        return x.flatten(1), fmaps


class ScaleDiscriminator(nn.Module):
    def __init__(self, width: int = 16, use_spectral_norm: bool = False):
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        w = width
        specs = [(1, w, 15, 1, 1), (w, w * 4, 41, 4, 4), (w * 4, w * 16, 41, 4, 16), (w * 16, w * 64, 41, 4, 16),
                 (w * 64, w * 64, 41, 4, 16), (w * 64, w * 64, 41, 1, 16), (w * 64, w * 64, 5, 1, 1)]  # fmt: skip
        self.convs = nn.ModuleList(
            norm(nn.Conv1d(i, o, k, s, padding=k // 2, groups=min(g, i, o))) for i, o, k, s, g in specs
        )
        self.conv_post = norm(nn.Conv1d(w * 64, 1, 3, 1, 1))

    def forward(self, x):
        fmaps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU)
            fmaps.append(x)
        x = self.conv_post(x)
        fmaps.append(x)
        return x.flatten(1), fmaps


class Discriminators(nn.Module):
    def __init__(self, periods: Sequence[int] = (2, 3, 5, 7, 11), mpd_width: int = 32, msd_width: int = 16):
        super().__init__()
        self.mpd = nn.ModuleList(PeriodDiscriminator(p, mpd_width) for p in periods)
        self.msd = nn.ModuleList(ScaleDiscriminator(msd_width, use_spectral_norm=i == 0) for i in range(3))
        self.pools = nn.ModuleList(
            [
                nn.Identity(),
                nn.AvgPool1d(4, 2, 2),
                nn.Sequential(nn.AvgPool1d(4, 2, 2), nn.AvgPool1d(4, 2, 2)),
            ]
        )

    def forward(self, wav):
        outs, fmaps = [], []
        for d in self.mpd:
            o, f = d(wav)
            outs.append(o)
            fmaps.append(f)
        for pool, d in zip(self.pools, self.msd):
            o, f = d(pool(wav))
            outs.append(o)
            fmaps.append(f)
        return outs, fmaps

    @classmethod
    def tiny(cls) -> Discriminators:
        return cls(periods=(2, 3), mpd_width=4, msd_width=2)


def discriminator_loss(real_outs, fake_outs) -> torch.Tensor:
    return sum(torch.mean((1 - r) ** 2) + torch.mean(f**2) for r, f in zip(real_outs, fake_outs))


def generator_adversarial_loss(fake_outs) -> torch.Tensor:
    return sum(torch.mean((1 - f) ** 2) for f in fake_outs)


def feature_matching_loss(real_fmaps, fake_fmaps) -> torch.Tensor:
    return 2 * sum(
        torch.mean(torch.abs(r.detach() - f))
        for rs, fs in zip(real_fmaps, fake_fmaps)
        for r, f in zip(rs, fs)
    )


class VocoderLoss(nn.Module):
    """Bundles one GAN training step's losses so the training script stays short."""

    def __init__(self, mel_weight: float = 45.0):
        super().__init__()
        self.mel = LogMel(center=False)
        self.mel_weight = mel_weight

    def mel_l1(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        n = min(real.shape[-1], fake.shape[-1])
        return F.l1_loss(self.mel(fake[..., :n].squeeze(1)), self.mel(real[..., :n].squeeze(1)))

    def discriminator_step(self, disc: Discriminators, real, fake) -> torch.Tensor:
        r, _ = disc(real)
        f, _ = disc(fake.detach())
        return discriminator_loss(r, f)

    def generator_step(self, disc: Discriminators, real, fake) -> tuple[torch.Tensor, dict]:
        _, r_maps = disc(real)
        f_out, f_maps = disc(fake)
        adv = generator_adversarial_loss(f_out)
        fm = feature_matching_loss(r_maps, f_maps)
        mel = self.mel_l1(real, fake)
        total = adv + fm + self.mel_weight * mel
        return total, {"g_adv": adv.detach(), "g_fm": fm.detach(), "mel_l1": mel.detach()}

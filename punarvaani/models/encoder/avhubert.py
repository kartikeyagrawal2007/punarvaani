"""AV-HuBERT-style visual speech encoder (stage [2]).

AV-HuBERT (Shi et al., ICLR 2022) is pretrained by *masked prediction*: mask
spans of the audio-visual input and predict cluster IDs of the audio for those
frames. Because the targets come from audio, the visual branch is forced to
learn which lip movements correspond to which speech sounds - exactly what lip
reading needs - without any transcripts. We reuse that English-pretrained model
and adapt it to Hindi instead of training from scratch (which would need
thousands of hours and GPUs we don't have).

Layout (names follow fairseq so ``load_fairseq_checkpoint`` maps by name)::

    video (B,1,T,88,88) -> feature_extractor_video.resnet -> (B,T,512) -> .proj -> (B,T,D)
    audio (B,T,104)     -> feature_extractor_audio.proj                -> (B,T,D)  [zeros if absent]
    concat (B,T,2D) -> layer_norm -> post_extract_proj (2D->D) -> [time mask] -> transformer -> (B,T,D)

For lip-to-speech the audio branch is fed zeros, which is precisely how
AV-HuBERT is evaluated for visual-only speech recognition (it was trained with
modality dropout, so it has seen "video only" inputs). Keeping the audio branch
lets the data engine optionally train audio-visually, and keeps checkpoints
loadable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from punarvaani.models.encoder.resnet import ResEncoder
from punarvaani.models.encoder.transformer import TransformerEncoder

# Pixel normalisation used by AV-HuBERT (mean/std of LRS3 grayscale mouth crops).
VIDEO_MEAN, VIDEO_STD = 0.421, 0.165


@dataclass
class EncoderConfig:
    dim: int = 1024
    layers: int = 24
    heads: int = 16
    ffn_dim: int = 4096
    layer_norm_first: bool = True
    resnet_widths: tuple[int, ...] = (64, 128, 256, 512)
    audio_feat_dim: int = 104  # 26 log filterbanks x 4 stacked frames
    conv_pos: int = 128
    conv_pos_groups: int = 16
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    layerdrop: float = 0.0
    # Time masking during fine-tuning (a SpecAugment analogue for video features).
    mask_prob: float = 0.0
    mask_length: int = 5
    extra: dict = field(default_factory=dict)

    @classmethod
    def preset(cls, name: str, **overrides) -> EncoderConfig:
        presets = {
            # Shapes of the released checkpoints.
            "large": dict(dim=1024, layers=24, heads=16, ffn_dim=4096, layer_norm_first=True),
            "base": dict(dim=768, layers=12, heads=12, ffn_dim=3072, layer_norm_first=False),
            # Small enough to train on a laptop CPU (tests, synthetic demo).
            "tiny": dict(dim=64, layers=2, heads=4, ffn_dim=128, resnet_widths=(8, 16, 32, 64), conv_pos=16,
                         conv_pos_groups=4, dropout=0.0, attention_dropout=0.0),
            "small": dict(dim=256, layers=6, heads=4, ffn_dim=1024, resnet_widths=(32, 64, 128, 256), conv_pos=32,
                          conv_pos_groups=8),
        }  # fmt: skip
        if name not in presets:
            raise KeyError(f"unknown preset {name!r}; choose from {sorted(presets)}")
        return cls(**{**presets[name], **overrides})


class SubModel(nn.Module):
    def __init__(self, resnet: nn.Module | None, input_dim: int, dim: int):
        super().__init__()
        self.resnet = resnet
        self.proj = nn.Linear(input_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.resnet is not None:
            x = self.resnet(x)
        return self.proj(x)


def compute_time_mask(batch: int, length: int, prob: float, span: int, padding_mask: torch.Tensor | None,
                      device: torch.device) -> torch.Tensor:  # fmt: skip
    """Bool (B, T) mask choosing about ``prob`` of the real frames in spans of ``span``."""
    mask = torch.zeros(batch, length, dtype=torch.bool, device=device)
    lengths = (
        (~padding_mask).sum(1) if padding_mask is not None else torch.full((batch,), length, device=device)
    )
    for b in range(batch):
        n = int(lengths[b])
        n_spans = int(prob * n / span + torch.rand(()).item())
        if n_spans == 0 or n <= span:
            continue
        starts = torch.randint(0, n - span, (n_spans,), device=device)
        idx = (starts[:, None] + torch.arange(span, device=device)).flatten()
        mask[b, idx] = True
    return mask


class VisualSpeechEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        resnet = ResEncoder(widths=cfg.resnet_widths)
        self.feature_extractor_video = SubModel(resnet, resnet.backend_out, cfg.dim)
        self.feature_extractor_audio = SubModel(None, cfg.audio_feat_dim, cfg.dim)
        self.layer_norm = nn.LayerNorm(2 * cfg.dim)
        self.post_extract_proj = nn.Linear(2 * cfg.dim, cfg.dim)
        self.mask_emb = nn.Parameter(torch.empty(cfg.dim).uniform_())
        self.encoder = TransformerEncoder(
            cfg.dim, cfg.layers, cfg.heads, cfg.ffn_dim, cfg.dropout, cfg.attention_dropout,
            cfg.activation_dropout, cfg.layer_norm_first, cfg.conv_pos, cfg.conv_pos_groups, cfg.layerdrop,
        )  # fmt: skip

    @property
    def dim(self) -> int:
        return self.cfg.dim

    def frontend_parameters(self):
        return self.feature_extractor_video.resnet.parameters()

    def forward(
        self,
        video: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
        return_all_layers: bool = False,
    ):
        """``video``: (B, T, H, W) or (B, 1, T, H, W), already normalised.

        Returns (B, T, D) features (and the per-layer list if requested).
        """
        if video.dim() == 4:
            video = video.unsqueeze(1)
        if padding_mask is not None:
            # Zero padded frames so that the 3D stem (5-frame kernel) sees exactly what
            # its own zero padding would show for an unpadded clip: batching must not
            # change the features of the last real frames.
            video = video.masked_fill(padding_mask[:, None, :, None, None], 0.0)
        v = self.feature_extractor_video(video)
        a = self.feature_extractor_audio(audio) if audio is not None else torch.zeros_like(v)
        x = self.post_extract_proj(self.layer_norm(torch.cat([a, v], dim=-1)))
        if self.training and self.cfg.mask_prob > 0:
            m = compute_time_mask(
                x.shape[0], x.shape[1], self.cfg.mask_prob, self.cfg.mask_length, padding_mask, x.device
            )
            x = torch.where(m.unsqueeze(-1), self.mask_emb.to(x.dtype), x)
        return self.encoder(x, padding_mask, return_all_layers=return_all_layers)

    def config_dict(self) -> dict:
        return asdict(self.cfg)


def normalize_video(video_uint8: torch.Tensor) -> torch.Tensor:
    return (video_uint8.float() / 255.0 - VIDEO_MEAN) / VIDEO_STD


# ----------------------------------------------------------------------------
# Loading Meta's released fairseq checkpoints
# ----------------------------------------------------------------------------

_PREFIXES = ("w2v_encoder.w2v_model.", "encoder.w2v_model.", "w2v_model.", "model.")


def convert_fairseq_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename fairseq AV-HuBERT keys to ours.

    Differences handled: wrapper prefixes from fine-tuned checkpoints; the
    positional conv's old-style weight norm (``weight_g``/``weight_v``) versus
    PyTorch's parametrisation (``original0``/``original1``). Pretraining-only
    tensors (``final_proj``, ``label_embs_concat``) are dropped.
    """
    out = {}
    for key, value in state.items():
        for p in _PREFIXES:
            if key.startswith(p):
                key = key[len(p) :]
                break
        if key.startswith(("final_proj", "label_embs_concat", "decoder.")):
            continue
        key = key.replace("pos_conv.0.weight_g", "pos_conv.conv.parametrizations.weight.original0")
        key = key.replace("pos_conv.0.weight_v", "pos_conv.conv.parametrizations.weight.original1")
        key = key.replace("pos_conv.0.bias", "pos_conv.conv.bias")
        out[key] = value
    return out


def load_fairseq_checkpoint(model: VisualSpeechEncoder, path: str | Path) -> dict[str, list[str]]:
    """Load AV-HuBERT weights (e.g. ``large_vox_iter5.pt``) into ``model``.

    Fairseq checkpoints pickle their config, so this needs ``weights_only=False``:
    only load files you downloaded from the official source. Returns the keys
    that did not match, so the caller can check nothing important was missed.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = convert_fairseq_state_dict(ckpt.get("model", ckpt))
    own = model.state_dict()
    shape_mismatch = [k for k, v in state.items() if k in own and own[k].shape != v.shape]
    for k in shape_mismatch:  # e.g. mask_emb sized for the input features in some versions
        state.pop(k)
    result = model.load_state_dict(state, strict=False)
    return {
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
        "shape_mismatch": shape_mismatch,
    }


def output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    """The encoder keeps the frame rate (25 Hz), so output length = input length."""
    return input_lengths


def pad_video_batch(videos: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([v.shape[0] for v in videos])
    t = int(lengths.max())
    batch = torch.stack([F.pad(v, (0, 0, 0, 0, 0, t - v.shape[0])) for v in videos])
    return batch, lengths

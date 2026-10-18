"""Transformer encoder in the fairseq wav2vec 2.0 / HuBERT style used by AV-HuBERT.

Two details that matter:

* **Convolutional positional embedding** instead of sinusoids: a wide grouped
  1-D conv (kernel 128 = 5 s at 25 fps) adds *relative* position information
  and generalises to any utterance length.
* **Separate q/k/v/out projections** (not PyTorch's fused ``in_proj_weight``),
  so names match fairseq checkpoints and LoRA can wrap exactly the projections
  we choose.

``layer_norm_first=True`` is pre-LN (AV-HuBERT Large); ``False`` is post-LN.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class MultiheadAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.heads, self.dropout = heads, dropout
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        return x.view(b, t, self.heads, d // self.heads).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """``key_padding_mask``: (B, S) bool, True = padded (ignored)."""
        key = query if key is None else key
        q, k, v = (
            self._split(self.q_proj(query)),
            self._split(self.k_proj(key)),
            self._split(self.v_proj(key)),
        )
        mask = None
        if key_padding_mask is not None:
            mask = ~key_padding_mask[:, None, None, :]  # SDPA: True = attend
        if causal:
            tq, tk = q.shape[2], k.shape[2]
            causal_mask = torch.ones(tq, tk, dtype=torch.bool, device=q.device).tril(tk - tq)
            mask = causal_mask if mask is None else mask & causal_mask
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        b, h, t, dh = out.shape
        return self.out_proj(out.transpose(1, 2).reshape(b, t, h * dh))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float = 0.1, attention_dropout: float = 0.1,
                 activation_dropout: float = 0.0, layer_norm_first: bool = True):  # fmt: skip
        super().__init__()
        self.self_attn = MultiheadAttention(dim, heads, attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, dim)
        self.final_layer_norm = nn.LayerNorm(dim)
        self.dropout, self.activation_dropout = dropout, activation_dropout
        self.layer_norm_first = layer_norm_first

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        x = F.dropout(F.gelu(self.fc1(x)), self.activation_dropout, self.training)
        return F.dropout(self.fc2(x), self.dropout, self.training)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.layer_norm_first:
            x = x + F.dropout(
                self.self_attn(self.self_attn_layer_norm(x), key_padding_mask=padding_mask),
                self.dropout,
                self.training,
            )
            return x + self._ffn(self.final_layer_norm(x))
        x = self.self_attn_layer_norm(
            x + F.dropout(self.self_attn(x, key_padding_mask=padding_mask), self.dropout, self.training)
        )
        return self.final_layer_norm(x + self._ffn(x))


class ConvPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, kernel: int = 128, groups: int = 16):
        super().__init__()
        conv = nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel // 2, groups=groups)
        std = (4 / (kernel * dim)) ** 0.5
        nn.init.normal_(conv.weight, mean=0, std=std)
        nn.init.zeros_(conv.bias)
        # Same normalisation (weight norm over dim=2) as fairseq, so g/v load directly.
        self.conv = weight_norm(conv, name="weight", dim=2)
        self.remove = 1 if kernel % 2 == 0 else 0  # "SamePad": even kernels produce one extra frame

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x.transpose(1, 2))
        if self.remove:
            y = y[..., : -self.remove]
        return F.gelu(y).transpose(1, 2)


class TransformerEncoder(nn.Module):
    def __init__(self, dim: int, layers: int, heads: int, ffn_dim: int, dropout: float = 0.1,
                 attention_dropout: float = 0.1, activation_dropout: float = 0.0, layer_norm_first: bool = True,
                 conv_pos: int = 128, conv_pos_groups: int = 16, layerdrop: float = 0.0):  # fmt: skip
        super().__init__()
        self.pos_conv = ConvPositionalEmbedding(dim, conv_pos, conv_pos_groups)
        self.layers = nn.ModuleList(
            TransformerEncoderLayer(
                dim, heads, ffn_dim, dropout, attention_dropout, activation_dropout, layer_norm_first
            )
            for _ in range(layers)
        )
        self.layer_norm = nn.LayerNorm(dim)
        self.layer_norm_first, self.dropout, self.layerdrop = layer_norm_first, dropout, layerdrop

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None, return_all_layers: bool = False
    ):
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)  # padding must not leak through the conv
        x = x + self.pos_conv(x)
        if not self.layer_norm_first:
            x = self.layer_norm(x)
        x = F.dropout(x, self.dropout, self.training)
        hidden = []
        for layer in self.layers:
            # LayerDrop (Fan et al. 2019): randomly skip whole layers while training.
            if self.training and self.layerdrop > 0 and torch.rand(()) < self.layerdrop:
                hidden.append(x)
                continue
            x = layer(x, padding_mask)
            hidden.append(x)
        if self.layer_norm_first:
            x = self.layer_norm(x)
        return (x, hidden) if return_all_layers else x

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from typing import Tuple, Optional

# =============================
# Transformer blocks (ViT style)
# =============================

class ViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, N, D); key_padding_mask: (B, N) with True = ignore
        h = self.norm1(x)
        attn_out, _ = self.attn(
            h, h, h, need_weights=False, key_padding_mask=key_padding_mask
        )  # (B, N, D)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ======================================
# Spatiotemporal ViT with tubelet embed
# ======================================

class VideoViTMasked(nn.Module):
    """
    Spatiotemporal ViT with Conv3d tubelet embedding.
    - Keeps (or restores) full token sequence length so it matches VideoMAE-style N = (T/tubelet)*(H/ps)*(W/ps).
    - Set mask_ratio=0.0 to keep N fixed (e.g., 1568 for T=16, tubelet=2, H=W=224, patch=16).

    Input:  (B, C, T, H, W)
    Output: (B, N, D)
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,        # <<< important to match VideoMAE (16 frames -> T' = 8)
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        mask_ratio: float = 0.0,      # 0.0 keeps N fixed; >0.0 masks then restores with zeros
        spatial_pool_tokens: Optional[Tuple[int, int]] = None,  # pool (Hp, Wp) before attention
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size   = patch_size
        self.tubelet_size = tubelet_size
        self.embed_dim    = embed_dim
        self.mask_ratio   = mask_ratio
        self.spatial_pool_tokens = spatial_pool_tokens

        # Tubelet patchify: (B, C, T, H, W) -> (B, D, T', H', W')
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
            bias=True
        )

        # Base grid dimensions (for standard 16 frames at 224x224, tubelet 2, patch 16)
        self.base_num_frames = 16
        self.base_Tp = self.base_num_frames // tubelet_size  # 8
        self.base_Hp = img_size // patch_size                 # 14
        self.base_Wp = img_size // patch_size                 # 14
        base_num_patches = self.base_Tp * self.base_Hp * self.base_Wp  # 1568

        # Positional embeddings: pre-initialized as nn.Parameter for optimizer registration
        self.pos_embed = nn.Parameter(torch.zeros(1, base_num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv3d)):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if getattr(m, "bias", None) is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def get_interpolated_pos_embed(self, Tp: int, Hp: int, Wp: int) -> torch.Tensor:
        """
        Space-time aware positional embedding interpolation:
        Preserves the (Hp, Wp) spatial grid while interpolating along the temporal axis Tp.
        Shape transformation: (1, N_base, D) -> (1, D, Tp_base, Hp_base, Wp_base) -> interpolate -> (1, N_new, D)
        """
        if Tp == self.base_Tp and Hp == self.base_Hp and Wp == self.base_Wp:
            return self.pos_embed

        D = self.embed_dim
        pos = self.pos_embed.reshape(1, self.base_Tp, self.base_Hp, self.base_Wp, D)
        pos = pos.permute(0, 4, 1, 2, 3)  # (1, D, Tp_base, Hp_base, Wp_base)

        # Spatial interpolation is independent of the temporal length.
        if Hp != self.base_Hp or Wp != self.base_Wp:
            pos = F.interpolate(pos, size=(self.base_Tp, Hp, Wp), mode='trilinear', align_corners=False)

        # Prefix semantics for shorter clips: a padded sample must see exactly
        # the same positional embeddings as the same sample run alone.
        if Tp <= self.base_Tp:
            pos = pos[:, :, :Tp]
        else:
            pos = F.interpolate(pos, size=(Tp, Hp, Wp), mode='trilinear', align_corners=False)

        # (1, D, Tp, Hp, Wp) -> (1, Tp * Hp * Wp, D)
        return pos.permute(0, 2, 3, 4, 1).flatten(1, 3)

    @torch.no_grad()
    def _rand_mask_idx(self, B: int, N: int, device: torch.device):
        K = max(1, int(round(N * (1.0 - self.mask_ratio))))
        perm = torch.argsort(torch.rand(B, N, device=device), dim=1)
        kept_idx = perm[:, :K]  # (B, K)
        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, kept_idx, False)
        return kept_idx, mask

    def forward(
        self,
        x: torch.Tensor,
        video_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, C, T, H, W)
        video_lengths: optional (B,) number of valid frames per sample.
        returns tokens: (B, N, D) where N = Tp * Hp' * Wp'
        """
        B, C, T, H, W = x.shape
        device = x.device

        # Zero padded frames BEFORE the 3D stem so a padded sample is equivalent
        # to running the shorter sample alone (the stem has a temporal kernel).
        frame_token_mask = None
        if video_lengths is not None and T > 0:
            lengths = video_lengths.to(device=device, dtype=torch.long).view(B, 1)
            frame_valid = torch.arange(T, device=device).unsqueeze(0) < lengths  # (B, T)
            x = x * frame_valid.view(B, 1, T, 1, 1).to(dtype=x.dtype)

        # Tubelet patchify
        x = self.proj(x)                  # (B, D, T', H', W')
        Tp, Hp, Wp = x.shape[-3], x.shape[-2], x.shape[-1]

        if self.spatial_pool_tokens is not None:
            ph, pw = self.spatial_pool_tokens
            ph, pw = min(ph, Hp), min(pw, Wp)
            x = F.adaptive_avg_pool3d(x, (Tp, ph, pw))
            Hp, Wp = ph, pw

        x = x.flatten(2).transpose(1, 2)  # (B, N, D), N = Tp*Hp*Wp
        N = x.size(1)
        Sp = Hp * Wp

        # Valid tubelets: a tubelet is valid if it contains at least one real frame.
        token_valid = None
        if video_lengths is not None:
            valid_tubelets = torch.div(
                video_lengths.to(device=device, dtype=torch.long) + self.tubelet_size - 1,
                self.tubelet_size,
                rounding_mode="floor",
            ).clamp(min=1, max=Tp)  # (B,)
            tubelet_valid = (
                torch.arange(Tp, device=device).unsqueeze(0) < valid_tubelets.unsqueeze(1)
            )  # (B, Tp)
            token_valid = tubelet_valid.unsqueeze(-1).expand(B, Tp, Sp).reshape(B, N)
            frame_token_mask = ~token_valid  # MultiheadAttention: True = ignore

        # Retrieve space-time interpolated positional embeddings
        pos_embed = self.get_interpolated_pos_embed(Tp, Hp, Wp).to(device=device, dtype=x.dtype)
        tokens = x + pos_embed  # (B, N, D)

        # Masking: ONLY applied during training if mask_ratio > 0.0
        apply_mask = self.training and (self.mask_ratio > 0.0)
        if apply_mask:
            kept_idx, mask = self._rand_mask_idx(B, N, device)
            batch_idx = torch.arange(B, device=device).unsqueeze(-1)
            tokens_vis = tokens[batch_idx, kept_idx]  # (B, K, D)
            vis_padding_mask = None
            if frame_token_mask is not None:
                vis_padding_mask = frame_token_mask[batch_idx, kept_idx]
        else:
            tokens_vis = tokens
            kept_idx = None
            vis_padding_mask = frame_token_mask

        # Encode
        for blk in self.blocks:
            tokens_vis = blk(tokens_vis, key_padding_mask=vis_padding_mask)
        tokens_vis = self.norm(tokens_vis)  # (B, K or N, D)

        if apply_mask:
            # Restore masked positions with zeros
            out = torch.zeros(B, N, self.embed_dim, device=device, dtype=tokens_vis.dtype)
            out[torch.arange(B, device=device).unsqueeze(-1), kept_idx] = tokens_vis
        else:
            out = tokens_vis  # (B, N, D)

        if token_valid is not None:
            out = out.masked_fill(~token_valid.unsqueeze(-1), 0.0)
        return out


# ==========================================
# Spatial Aggregation & Temporal Adapter
# ==========================================

class SpatialAttentionPool(nn.Module):
    """
    Collapses Sp spatial tokens into 1 feature vector per temporal tubelet:
    (B, Tp, Sp, D) -> (B, Tp, D)
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, 1, embed_dim) * 0.02)
        self.scale = embed_dim ** -0.5

    def forward(self, x: torch.Tensor, tubelet_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, Tp, Sp, D); tubelet_mask: (B, Tp) bool, False = padded tubelet
        scores = torch.matmul(x, self.query.transpose(-1, -2)) * self.scale
        attn = torch.softmax(scores, dim=2)
        out = (x * attn).sum(dim=2)  # (B, Tp, D)
        if tubelet_mask is not None:
            out = out * tubelet_mask.unsqueeze(-1).to(dtype=out.dtype)
        return out


class TemporalAdapter(nn.Module):
    """
    Configurable temporal adapter module:
    Reduces channel dimension and performs learned temporal upsampling along the time axis:
    (B, in_channels, Tp) -> (B, adapter_dim, T_ctc) where T_ctc = Tp * upsample_factor
    """
    def __init__(self, in_channels: int = 768, adapter_dim: int = 256, upsample_factor: int = 4):
        super().__init__()
        self.upsample_factor = upsample_factor
        self.conv_in = nn.Sequential(
            nn.Conv1d(in_channels, adapter_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(adapter_dim),
            nn.ReLU(inplace=True)
        )
        if upsample_factor > 1:
            self.upsample = nn.ConvTranspose1d(
                adapter_dim, adapter_dim,
                kernel_size=upsample_factor,
                stride=upsample_factor,
                padding=0
            )
        else:
            self.upsample = nn.Identity()

        self.refine = nn.Sequential(
            nn.Conv1d(adapter_dim, adapter_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(adapter_dim),
            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, in_channels, Tp)
        valid_lengths: optional (B,) number of valid *tubelets*. Padded steps are
        zeroed before the refine convolution so a padded batch matches a shorter
        standalone clip (the refine conv uses implicit zero padding).
        """
        h = self.conv_in(x)
        h = self.upsample(h)

        step_mask = None
        if valid_lengths is not None:
            steps = torch.clamp(
                valid_lengths.to(h.device).long() * self.upsample_factor,
                max=h.size(2),
            )
            step_mask = (
                torch.arange(h.size(2), device=h.device).unsqueeze(0) < steps.unsqueeze(1)
            )
            h = h * step_mask.unsqueeze(1).to(dtype=h.dtype)

        h = self.refine(h)

        if step_mask is not None:
            h = h * step_mask.unsqueeze(1).to(dtype=h.dtype)
        return h  # (B, adapter_dim, Tp * upsample_factor)


# ==========================================
# 3D-CNN + ResNet-18 lip front end
# ==========================================

class _BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + (x if self.down is None else self.down(x)), inplace=True)


class LipResNetFrontend(nn.Module):
    """
    The standard lip-reading front end: one 3D convolution over 5 frames, then a 2D ResNet-18 per frame.
    CNNs already know about edges and local shapes, so they learn lip shapes from far less data than a ViT.

    Input  : (B, C, T, H, W) RGB in [0, 1] (any H, W; usually 224 mouth crops)
    Output : (B, T, 512) one feature vector per frame (padded frames are zero)

    Frames are turned grey, resized to 96x96 and cropped to 88x88: a random crop plus a random horizontal
    flip per clip during training, the centre crop otherwise. Batch norm only ever sees real frames, so
    padding never changes a clip's features.
    """
    RESIZE = 96
    CROP = 88
    MEAN = 0.421
    STD = 0.165

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.stem_bn = nn.BatchNorm2d(64)
        layers, in_ch = [], 64
        for out_ch, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
            layers += [_BasicBlock(in_ch, out_ch, stride), _BasicBlock(out_ch, out_ch, 1)]
            in_ch = out_ch
        self.trunk = nn.Sequential(*layers)
        self.out_dim = 512

    def _prepare(self, x: torch.Tensor, frame_valid: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        if C == 3:
            weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1, 1)
            x = (x * weights).sum(dim=1, keepdim=True)
        elif C != 1:
            x = x.mean(dim=1, keepdim=True)
        frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, 1, H, W)
        if (H, W) != (self.RESIZE, self.RESIZE):
            frames = F.interpolate(frames, size=(self.RESIZE, self.RESIZE), mode="bilinear",
                                   align_corners=False, antialias=True)
        frames = frames.view(B, T, 1, self.RESIZE, self.RESIZE)

        margin = self.RESIZE - self.CROP
        if self.training:
            crops = []
            for b in range(B):  # one random crop and flip per clip, the same for all its frames
                top, left = (int(v) for v in torch.randint(0, margin + 1, (2,)))
                clip = frames[b, :, :, top:top + self.CROP, left:left + self.CROP]
                if torch.rand(()) < 0.5:
                    clip = clip.flip(-1)
                crops.append(clip)
            frames = torch.stack(crops)
        else:
            m = margin // 2
            frames = frames[:, :, :, m:m + self.CROP, m:m + self.CROP]

        frames = (frames - self.MEAN) / self.STD
        frames = frames * frame_valid.view(B, T, 1, 1, 1).to(frames.dtype)  # padded frames = 0, like conv padding
        return frames.permute(0, 2, 1, 3, 4)  # (B, 1, T, 88, 88)

    def forward(self, x: torch.Tensor, frame_valid: torch.Tensor) -> torch.Tensor:
        B, _, T = x.shape[:3]
        x = self.stem(self._prepare(x, frame_valid))     # (B, 64, T, 44, 44)
        x = x.permute(0, 2, 1, 3, 4)                      # (B, T, 64, 44, 44)
        valid = frame_valid.reshape(-1)
        real = x.reshape(B * T, *x.shape[2:])[valid]      # batch norm sees real frames only
        real = F.relu(self.stem_bn(real), inplace=True)
        real = F.max_pool2d(real, kernel_size=3, stride=2, padding=1)
        real = self.trunk(real)
        real = real.mean(dim=(2, 3))                      # (N_real, 512)
        out = real.new_zeros(B * T, self.out_dim)
        out[valid] = real
        return out.view(B, T, self.out_dim)


def sinusoidal_positions(length: int, dim: int, device, dtype) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe.to(dtype)


def stage1_kwargs_from_state_dict(state_dict: dict) -> dict:
    """Architecture settings that can be read off saved weights, so eval/infer rebuild the right model."""
    if not any(k.startswith("frontend.") for k in state_dict):
        return {"frontend": "vit"}
    layers = {int(k.split(".")[2]) for k in state_dict if k.startswith("backend.layers.")}
    return {
        "frontend": "resnet",
        "backend_dim": int(state_dict["frontend_proj.weight"].shape[0]),
        "backend_layers": len(layers),
    }


# ============================
# VALLR model (pure PyTorch)
# ============================

class VALLR(nn.Module):
    """
    VALLR with spatiotemporal ViT (tubelet embedding) + spatial attention pooling
    + configurable temporal adapter (learned upsampling) + adapter + CTC head.

    Returns (logits, adapted_features):
        logits:           (B, T_ctc, num_classes)
        adapted_features: (B, T_ctc, ctc_hidden_size)
    """
    def __init__(
        self,
        adapter_dim: int = 256,
        feature_size: int = 768,      # ViT embed dim
        ctc_hidden_size: int = 256,
        num_classes: int = 55,
        temporal_upsample_factor: int = 4,

        # ViT (video) config
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,        # with T=16 gives T' = 8
        in_channels: int = 3,
        vit_depth: int = 8,
        vit_heads: int = 12,
        vit_mlp_ratio: float = 4.0,
        vit_dropout: float = 0.2,
        vit_attn_dropout: float = 0.0,
        vit_mask_ratio: float = 0.0,  # 0.0 default for supervised mode
        spatial_pool_tokens: Optional[Tuple[int, int]] = None,

        # "vit" (spatiotemporal ViT above) or "resnet" (3D-CNN + ResNet-18 + temporal transformer)
        frontend: str = "vit",
        backend_dim: int = 256,
        backend_layers: int = 6,
        backend_dropout: float = 0.1,
    ):
        super().__init__()
        if frontend not in ("vit", "resnet"):
            raise ValueError(f"frontend must be 'vit' or 'resnet', got {frontend!r}")
        self.frontend_type = frontend
        self.temporal_upsample_factor = temporal_upsample_factor
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.img_size = img_size
        self.spatial_pool_tokens = spatial_pool_tokens

        if frontend == "resnet":
            if temporal_upsample_factor % tubelet_size != 0:
                raise ValueError("--temporal_upsample_factor must be a multiple of the tubelet size (2) for the resnet front end")
            # Per-frame features, so the adapter upsamples by factor/tubelet to keep the same CTC length
            # (ceil(frames / tubelet) * factor) that the dataset and collate already assume.
            self.frontend = LipResNetFrontend()
            self.frontend_proj = nn.Linear(self.frontend.out_dim, backend_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=backend_dim, nhead=max(1, backend_dim // 64), dim_feedforward=4 * backend_dim,
                dropout=backend_dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            self.backend = nn.TransformerEncoder(layer, num_layers=backend_layers, enable_nested_tensor=False)
            self.backend_norm = nn.LayerNorm(backend_dim)
            self.temporal_adapter = TemporalAdapter(
                in_channels=backend_dim, adapter_dim=adapter_dim,
                upsample_factor=temporal_upsample_factor // tubelet_size,
            )
            self.adapter = nn.Sequential(nn.Linear(adapter_dim, ctc_hidden_size), nn.ReLU(inplace=True))
            self.ctc_head = nn.Linear(ctc_hidden_size, num_classes)
            # Kaiming init for the CNN parts; the transformer and CTC head keep PyTorch's default init.
            for part in (self.frontend, self.temporal_adapter, self.adapter):
                part.apply(self._init_weights)
            return

        # 1) Spatiotemporal ViT backbone producing tokens (B, N, D)
        self.backbone = VideoViTMasked(
            img_size=img_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            in_chans=in_channels,
            embed_dim=feature_size,
            depth=vit_depth,
            num_heads=vit_heads,
            mlp_ratio=vit_mlp_ratio,
            dropout=vit_dropout,
            attn_dropout=vit_attn_dropout,
            mask_ratio=vit_mask_ratio,
            spatial_pool_tokens=spatial_pool_tokens,
        )

        # 2) Spatial Attention Pooling: (B, Tp, Sp, D) -> (B, Tp, D)
        self.spatial_pool = SpatialAttentionPool(embed_dim=feature_size)

        # 3) Temporal Adapter with configurable learned temporal upsampling: (B, D, Tp) -> (B, adapter_dim, T_ctc)
        self.temporal_adapter = TemporalAdapter(
            in_channels=feature_size,
            adapter_dim=adapter_dim,
            upsample_factor=temporal_upsample_factor
        )

        # 4) Adapter to CTC hidden size
        self.adapter = nn.Sequential(
            nn.Linear(adapter_dim, ctc_hidden_size),
            nn.ReLU(inplace=True),
        )

        # 5) CTC head
        self.ctc_head = nn.Linear(ctc_hidden_size, num_classes)

        # Init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d)):
            init.kaiming_normal_(module.weight, nonlinearity='relu')
            if getattr(module, "bias", None) is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            init.ones_(module.weight)
            init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            init.ones_(module.weight)
            init.zeros_(module.bias)

    @staticmethod
    def compute_input_lengths(
        video_lengths: torch.Tensor,
        tubelet_size: int,
        temporal_upsample_factor: int,
    ) -> torch.Tensor:
        """CTC time steps for each sample: ceil(frames / tubelet) * upsample."""
        tubelets = torch.div(
            video_lengths + tubelet_size - 1, tubelet_size, rounding_mode="floor"
        ).clamp(min=1)
        return tubelets * temporal_upsample_factor

    def forward(
        self,
        video_inputs: torch.Tensor,
        video_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Accepts either (B, C, T, H, W) or (B, T, C, H, W).
        video_lengths: optional (B,) valid frame counts for padded batches.
        Returns:
            logits  : (B, T_ctc, num_classes)
            adapted : (B, T_ctc, ctc_hidden_size)
        """
        # Normalize layout to (B, C, T, H, W)
        if video_inputs.ndim != 5:
            raise ValueError(f"Expected 5D input, got shape {video_inputs.shape}")

        if video_inputs.shape[1] in (1, 3):
            vid = video_inputs
        else:
            vid = video_inputs.permute(0, 2, 1, 3, 4).contiguous()

        B, C, T, H, W = vid.shape

        # Pad the temporal axis to a tubelet multiple so the final odd frame is
        # covered instead of silently dropped by the Conv3d stride.
        if T % self.tubelet_size != 0:
            pad = self.tubelet_size - (T % self.tubelet_size)
            vid = F.pad(vid, (0, 0, 0, 0, 0, pad))
            T += pad

        if self.frontend_type == "resnet":
            return self._forward_resnet(vid, video_lengths)

        Tp = T // self.tubelet_size
        Hp = H // self.patch_size
        Wp = W // self.patch_size
        if self.spatial_pool_tokens is not None:
            Hp = min(self.spatial_pool_tokens[0], Hp)
            Wp = min(self.spatial_pool_tokens[1], Wp)
        Sp = Hp * Wp

        device = vid.device
        if video_lengths is None:
            valid_tubelets = torch.full((B,), Tp, device=device, dtype=torch.long)
        else:
            video_lengths = video_lengths.to(device=device, dtype=torch.long)
            valid_tubelets = torch.div(
                video_lengths + self.tubelet_size - 1,
                self.tubelet_size,
                rounding_mode="floor",
            ).clamp(min=1, max=Tp)

        # 1) ViT backbone: tokens (B, N, D) where N = Tp * Sp
        tokens = self.backbone(vid, video_lengths=video_lengths)

        # 2) Unflatten to space-time grid: (B, Tp, Sp, D)
        grid_tokens = tokens.view(B, Tp, Sp, -1)

        # 3) Spatial Attention Pooling: (B, Tp, D)
        tubelet_mask = (
            torch.arange(Tp, device=device).unsqueeze(0) < valid_tubelets.unsqueeze(1)
        )  # (B, Tp)
        t_feats = self.spatial_pool(grid_tokens, tubelet_mask=tubelet_mask)

        # 4) Temporal Adapter (Learned Upsampling): (B, adapter_dim, T_ctc)
        t_feats = t_feats.transpose(1, 2)              # (B, D, Tp)
        adapted_temp = self.temporal_adapter(           # (B, adapter_dim, T_ctc)
            t_feats, valid_lengths=valid_tubelets
        )
        adapted_temp = adapted_temp.transpose(1, 2)    # (B, T_ctc, adapter_dim)

        # Zero padded CTC steps so padded batches match shorter standalone runs.
        t_ctc = adapted_temp.size(1)
        valid_steps = torch.clamp(
            valid_tubelets * self.temporal_upsample_factor, max=t_ctc
        )
        step_mask = torch.arange(t_ctc, device=device).unsqueeze(0) < valid_steps.unsqueeze(1)
        adapted_temp = adapted_temp * step_mask.unsqueeze(-1).to(dtype=adapted_temp.dtype)

        # 5) Adapter
        adapted = self.adapter(adapted_temp)           # (B, T_ctc, ctc_hidden_size)

        # 6) CTC logits
        logits = self.ctc_head(adapted)                # (B, T_ctc, num_classes)

        return logits, adapted

    def _forward_resnet(self, vid: torch.Tensor, video_lengths: Optional[torch.Tensor]):
        B, _, T = vid.shape[:3]  # T is already a tubelet multiple
        device = vid.device
        if video_lengths is None:
            frames = torch.full((B,), T, device=device, dtype=torch.long)
        else:
            frames = video_lengths.to(device=device, dtype=torch.long).clamp(min=1, max=T)
        # Count whole tubelets, as the CTC length formula does (an odd last frame rounds up).
        valid_tubelets = torch.div(frames + self.tubelet_size - 1, self.tubelet_size, rounding_mode="floor")
        valid_frames = (valid_tubelets * self.tubelet_size).clamp(max=T)
        real_frames = torch.arange(T, device=device).unsqueeze(0) < frames.unsqueeze(1)    # pixels to read
        step_frames = torch.arange(T, device=device).unsqueeze(0) < valid_frames.unsqueeze(1)  # steps to keep

        feats = self.frontend(vid, real_frames)                       # (B, T, 512)
        h = self.frontend_proj(feats)
        h = h + sinusoidal_positions(T, h.size(-1), device, h.dtype).unsqueeze(0)
        h = h * step_frames.unsqueeze(-1).to(h.dtype)
        h = self.backend(h, src_key_padding_mask=~step_frames)
        h = self.backend_norm(h) * step_frames.unsqueeze(-1).to(h.dtype)

        adapted_temp = self.temporal_adapter(h.transpose(1, 2), valid_lengths=valid_frames)  # (B, A, T*up)
        adapted_temp = adapted_temp.transpose(1, 2)
        t_ctc = adapted_temp.size(1)
        valid_steps = torch.clamp(valid_tubelets * self.temporal_upsample_factor, max=t_ctc)
        step_mask = torch.arange(t_ctc, device=device).unsqueeze(0) < valid_steps.unsqueeze(1)
        adapted_temp = adapted_temp * step_mask.unsqueeze(-1).to(adapted_temp.dtype)
        adapted = self.adapter(adapted_temp)
        return self.ctc_head(adapted), adapted

    @staticmethod
    def decode_ctc(logits: torch.Tensor, vocab: dict, blank_id: int = 0):
        """
        Greedy CTC decoding: collapses repeats and removes blank tokens.
        Args:
            logits: Tensor of shape (B, T, num_classes)
            vocab: Dict mapping phoneme str -> int ID
            blank_id: Integer token ID for blank (default: 0)
        Returns:
            List[str]: Decoded phoneme sequence per batch item e.g. ['N M S TD E', ...]
        """
        reverse_vocab = {v: k for k, v in vocab.items()}
        preds = torch.argmax(logits, dim=-1)  # (B, T)
        batch_results = []

        for b in range(preds.size(0)):
            seq = preds[b].tolist()
            decoded = []
            prev_token = None
            for token_id in seq:
                if token_id != blank_id and token_id != prev_token:
                    phoneme = reverse_vocab.get(token_id, "")
                    if phoneme and phoneme != "<pad>":
                        decoded.append(phoneme)
                prev_token = token_id
            batch_results.append(" ".join(decoded))

        return batch_results


# Alias for backward compatibility
ML_VALLR = VALLR

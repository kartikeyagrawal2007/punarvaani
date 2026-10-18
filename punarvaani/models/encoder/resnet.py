"""Visual front-end: 3D conv stem + 2D ResNet-18 trunk, applied to every frame.

This is the standard lip-reading front-end (Stafylakis & Tzimiropoulos 2017,
Martinez et al. 2020) and the exact one AV-HuBERT uses. Module and parameter
names match ``avhubert/resnet.py`` so pretrained weights load by name.

Shapes: (B, 1, T, 88, 88) -> stem (5x7x7 conv, stride 1x2x2) -> (B, 64, T, 44, 44)
-> max-pool -> (B, 64, T, 22, 22) -> fold time into batch -> ResNet-18 ->
global average pool -> (B, T, 512).

The 3D stem's temporal kernel of 5 frames (200 ms) is what lets the network see
*motion*, not just mouth shape: a single frame cannot tell an opening mouth
from a closing one. Everything after it is per-frame; temporal modelling is
left to the transformer.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _act(kind: str, channels: int) -> nn.Module:
    return nn.PReLU(num_parameters=channels) if kind == "prelu" else nn.ReLU(inplace=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        relu_type: str = "prelu",
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = _act(relu_type, planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.relu2 = _act(relu_type, planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu2(out + residual)


class ResNet(nn.Module):
    def __init__(
        self,
        layers: Sequence[int] = (2, 2, 2, 2),
        widths: Sequence[int] = (64, 128, 256, 512),
        relu_type: str = "prelu",
    ):
        super().__init__()
        self.inplanes = widths[0]
        self.relu_type = relu_type
        self.layer1 = self._make_layer(widths[0], layers[0], stride=1)
        self.layer2 = self._make_layer(widths[1], layers[1], stride=2)
        self.layer3 = self._make_layer(widths[2], layers[2], stride=2)
        self.layer4 = self._make_layer(widths[3], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = widths[3]

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample, self.relu_type)]
        self.inplanes = planes
        layers += [BasicBlock(planes, planes, relu_type=self.relu_type) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.avgpool(x).flatten(1)


class ResEncoder(nn.Module):
    def __init__(self, widths: Sequence[int] = (64, 128, 256, 512), relu_type: str = "prelu"):
        super().__init__()
        stem = widths[0]
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, stem, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(stem),
            _act(relu_type, stem),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.trunk = ResNet(widths=widths, relu_type=relu_type)
        self.backend_out = self.trunk.out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, T, H, W) -> (B, T, C)."""
        b = x.shape[0]
        x = self.frontend3D(x)
        t = x.shape[2]
        x = x.transpose(1, 2).reshape(b * t, x.shape[1], x.shape[3], x.shape[4])
        return self.trunk(x).view(b, t, -1)

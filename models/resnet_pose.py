"""Naive ResNet18 baseline: thermal map → 21×3 hand joints."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNetPose(nn.Module):
    """Single-frame thermal → joints; no physics / depth / recon."""

    has_physics = False

    def __init__(self, pretrained: bool = True, **_kwargs):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        old = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            1,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                backbone.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(feat_dim, 21 * 3)

    def forward(self, thermal, ambient=None):
        x = thermal
        if x.dim() == 4 and x.shape[1] != 1:
            x = x[:, -1:].contiguous()
        feat = self.backbone(x)
        joints = self.head(feat).view(-1, 21, 3)
        return {"joints": joints, "depth": None, "t_recon": None, "mask_logits": None}

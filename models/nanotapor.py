"""NanoTapor / EdgeTapor student (official TAPOR lightweight CNN)."""
from __future__ import annotations

import torch.nn as nn


class NanoTapor(nn.Module):
    """4-conv CNN + FC student; matches EdgeTaporStudent in NanoTapor_train.py."""

    has_physics = False

    def __init__(self, feat_dim: int = 336, last_conv_channel: int = 64, **_kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, last_conv_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(last_conv_channel * 3 * 4, feat_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(nn.Linear(feat_dim, 21 * 3))

    def forward(self, thermal, ambient=None):
        # thermal: B,1,H,W (32x24)
        x = thermal
        if x.dim() == 4 and x.shape[1] != 1:
            x = x[:, -1:].contiguous()  # take current if temporal
        feat = self.encoder(x)
        joints = self.decoder(feat).view(-1, 21, 3)
        return {"joints": joints, "depth": None, "t_recon": None, "mask_logits": None}

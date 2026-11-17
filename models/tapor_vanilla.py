"""Compact vanilla TAPOR: Thermo-Space → KP → Retrieval."""
from __future__ import annotations

import torch.nn as nn

from .common import KPFeatureExtractor, PositionRetrievalDecoder, ThermoSpaceEncoder


class TaporVanilla(nn.Module):
    has_physics = False

    def __init__(self, in_ch=1, spatial_ch=64, kp_dim=48):
        super().__init__()
        self.thermo = ThermoSpaceEncoder(in_ch, spatial_ch)
        self.kp = KPFeatureExtractor(in_ch, kp_dim)
        self.decoder = PositionRetrievalDecoder(spatial_ch, kp_dim)

    def forward(self, thermal, ambient=None):
        spatial = self.thermo(thermal)
        kp = self.kp(thermal)
        joints = self.decoder(spatial, kp)
        return {"joints": joints, "depth": None, "t_recon": None, "mask_logits": None}

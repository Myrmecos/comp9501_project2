"""Structure B: Shared Thermo-Space trunk → primitive heads + KP/retrieval."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBNReLU, FiLM, KPFeatureExtractor, PositionRetrievalDecoder, ResidualConv, ThermoSpaceEncoder
from .forward_thermal import ForwardThermal
from .physics_utils import ambient_normalize, maybe_detach


class HybridB(nn.Module):
    has_physics = True

    def __init__(
        self, in_ch=1, spatial_ch=128, kp_dim=96, detach_physics_for_pose=True, use_forward=True, **_kwargs
    ):
        super().__init__()
        self.detach_physics_for_pose = bool(detach_physics_for_pose)
        self.use_forward = bool(use_forward)
        self.thermo = ThermoSpaceEncoder(in_ch, spatial_ch)
        self.prim_head = nn.Sequential(
            ResidualConv(spatial_ch),
            ConvBNReLU(spatial_ch, spatial_ch),
            nn.Conv2d(spatial_ch, 5, 1),
        )
        self.depth_scale = nn.Parameter(torch.tensor(0.6))
        self.depth_bias = nn.Parameter(torch.tensor(0.25))
        self.forward_model = ForwardThermal() if self.use_forward else None
        self.film = FiLM(cue_dim=4, ch=spatial_ch)
        self.kp = KPFeatureExtractor(in_ch=3, feat_dim=kp_dim)
        self.decoder = PositionRetrievalDecoder(spatial_ch, kp_dim)

    def forward(self, thermal, ambient):
        thermal_n = ambient_normalize(thermal, ambient)
        spatial0 = self.thermo(thermal_n)
        raw = self.prim_head(spatial0)
        raw = F.interpolate(raw, size=thermal.shape[-2:], mode="bilinear", align_corners=False)
        depth = (self.depth_bias.abs() + self.depth_scale.abs() * torch.sigmoid(raw[:, 0:1])).clamp(0.05, 1.5)
        eps = torch.sigmoid(raw[:, 1:2]) * 0.9 + 0.05
        tt = raw[:, 2:3] * 8.0 + 32.0
        refl = F.softplus(raw[:, 3:4])
        mask_logits = raw[:, 4:5]
        t_recon = None
        if self.forward_model is not None:
            t_recon = self.forward_model(depth, eps, tt, refl, ambient)
        d = maybe_detach(depth, self.detach_physics_for_pose)
        e = maybe_detach(eps, self.detach_physics_for_pose)
        cue = torch.cat(
            [d.mean(dim=(2, 3)), e.mean(dim=(2, 3)), tt.mean(dim=(2, 3)) / 50.0, ambient],
            dim=1,
        )
        spatial = self.film(spatial0, cue)
        kp_in = torch.cat([thermal_n, d, e], dim=1)
        kp = self.kp(kp_in)
        joints = self.decoder(spatial, kp)
        return {
            "joints": joints,
            "depth": depth,
            "t_recon": t_recon,
            "mask_logits": mask_logits,
            "emissivity": eps,
            "surface_temp": tt,
            "reflection": refl,
        }

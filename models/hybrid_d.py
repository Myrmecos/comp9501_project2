"""Structure D: Physics maps condition f_spatial; KP queries that space."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBNReLU, FiLM, InverseUNet, KPFeatureExtractor, PositionRetrievalDecoder, ResidualConv
from .forward_thermal import ForwardThermal
from .physics_utils import ambient_normalize, maybe_detach


class HybridD(nn.Module):
    has_physics = True

    def __init__(
        self,
        in_ch=1,
        spatial_ch=128,
        kp_dim=96,
        inverse_base=64,
        detach_physics_for_pose=True,
        use_forward=True,
        **_kwargs,
    ):
        super().__init__()
        self.detach_physics_for_pose = bool(detach_physics_for_pose)
        self.use_forward = bool(use_forward)
        self.inverse = InverseUNet(in_ch=in_ch, base=inverse_base, out_ch=5)
        self.forward_model = ForwardThermal() if self.use_forward else None
        self.spatial_builder = nn.Sequential(
            ConvBNReLU(5, 64),
            ResidualConv(64),
            ConvBNReLU(64, spatial_ch),
            ResidualConv(spatial_ch),
        )
        self.inv_proj = nn.Conv2d(inverse_base, spatial_ch, 1)
        self.film = FiLM(cue_dim=4, ch=spatial_ch)
        self.kp = KPFeatureExtractor(in_ch=2, feat_dim=kp_dim)
        self.decoder = PositionRetrievalDecoder(spatial_ch, kp_dim)

    def forward(self, thermal, ambient):
        thermal_n = ambient_normalize(thermal, ambient)
        prim = self.inverse(thermal_n)
        t_recon = None
        if self.forward_model is not None:
            t_recon = self.forward_model(
                prim["depth"], prim["emissivity"], prim["surface_temp"], prim["reflection"], ambient
            )
        mask = torch.sigmoid(prim["mask_logits"])
        d = maybe_detach(prim["depth"], self.detach_physics_for_pose)
        e = maybe_detach(prim["emissivity"], self.detach_physics_for_pose)
        tt = maybe_detach(prim["surface_temp"], self.detach_physics_for_pose)
        m = maybe_detach(mask, self.detach_physics_for_pose)
        packed = torch.cat([thermal_n, d, e, tt / 50.0, m], dim=1)
        spatial = self.spatial_builder(packed)
        inv_feat = maybe_detach(prim["feat"], self.detach_physics_for_pose)
        if inv_feat.shape[-2:] != spatial.shape[-2:]:
            inv_feat = F.interpolate(inv_feat, size=spatial.shape[-2:], mode="bilinear", align_corners=False)
        spatial = spatial + self.inv_proj(inv_feat)
        cue = torch.cat(
            [d.mean(dim=(2, 3)), e.mean(dim=(2, 3)), tt.mean(dim=(2, 3)) / 50.0, ambient],
            dim=1,
        )
        spatial = self.film(spatial, cue)
        kp = self.kp(torch.cat([thermal_n, d], dim=1))
        joints = self.decoder(spatial, kp)
        return {
            "joints": joints,
            "depth": prim["depth"],
            "t_recon": t_recon,
            "mask_logits": prim["mask_logits"],
            "emissivity": prim["emissivity"],
            "surface_temp": prim["surface_temp"],
            "reflection": prim["reflection"],
        }

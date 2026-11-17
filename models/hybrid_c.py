"""Structure C: Parallel physics + TAPOR paths with gated late fusion on joints."""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import InverseUNet, KPFeatureExtractor, PositionRetrievalDecoder, ThermoSpaceEncoder
from .forward_thermal import ForwardThermal
from .physics_utils import ambient_normalize, maybe_detach


class HybridC(nn.Module):
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
        self.thermo = ThermoSpaceEncoder(in_ch, spatial_ch)
        self.kp = KPFeatureExtractor(in_ch, kp_dim)
        self.decoder = PositionRetrievalDecoder(spatial_ch, kp_dim)
        self.fuse = nn.Sequential(
            nn.Linear(21 * 3 + 6, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 21 * 3 + 1),
        )

    def forward(self, thermal, ambient):
        thermal_n = ambient_normalize(thermal, ambient)
        prim = self.inverse(thermal_n)
        t_recon = None
        if self.forward_model is not None:
            t_recon = self.forward_model(
                prim["depth"], prim["emissivity"], prim["surface_temp"], prim["reflection"], ambient
            )
        spatial = self.thermo(thermal_n)
        kp = self.kp(thermal_n)
        joints0 = self.decoder(spatial, kp)
        mask = torch.sigmoid(prim["mask_logits"])
        d = maybe_detach(prim["depth"], self.detach_physics_for_pose)
        e = maybe_detach(prim["emissivity"], self.detach_physics_for_pose)
        tt = maybe_detach(prim["surface_temp"], self.detach_physics_for_pose)
        m = maybe_detach(mask, self.detach_physics_for_pose)
        d_mean = d.mean(dim=(2, 3))
        d_masked = (d * m).sum(dim=(2, 3)) / m.sum(dim=(2, 3)).clamp(min=1.0)
        d_max = d.amax(dim=(2, 3))
        eps_mean = e.mean(dim=(2, 3))
        tt_mean = tt.mean(dim=(2, 3)) / 50.0
        cue = torch.cat([d_mean, d_masked, d_max, eps_mean, tt_mean, ambient], dim=1)
        x = torch.cat([joints0.flatten(1), cue], dim=1)
        out = self.fuse(x)
        delta = out[:, : 21 * 3].view(-1, 21, 3)
        gate = torch.sigmoid(out[:, -1:]).view(-1, 1, 1)
        joints = joints0 + gate * delta
        return {
            "joints": joints,
            "depth": prim["depth"],
            "t_recon": t_recon,
            "mask_logits": prim["mask_logits"],
            "emissivity": prim["emissivity"],
            "surface_temp": prim["surface_temp"],
            "reflection": prim["reflection"],
        }

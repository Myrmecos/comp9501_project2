"""Structure A: Inverse+Forward first, then TAPOR retrieval on packed maps."""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import FiLM, InverseUNet, KPFeatureExtractor, PositionRetrievalDecoder, ThermoSpaceEncoder
from .forward_thermal import ForwardThermal
from .physics_utils import ambient_normalize, maybe_detach


class HybridA(nn.Module):
    has_physics = True

    def __init__(
        self, in_ch=1, spatial_ch=128, kp_dim=96, inverse_base=64, detach_physics_for_pose=True, **_kwargs
    ):
        super().__init__()
        self.detach_physics_for_pose = bool(detach_physics_for_pose)
        self.inverse = InverseUNet(in_ch=in_ch, base=inverse_base, out_ch=5)
        self.forward_model = ForwardThermal()
        self.thermo = ThermoSpaceEncoder(in_ch=4, out_ch=spatial_ch)
        self.film = FiLM(cue_dim=4, ch=spatial_ch)
        self.kp = KPFeatureExtractor(in_ch=4, feat_dim=kp_dim)
        self.decoder = PositionRetrievalDecoder(spatial_ch, kp_dim)

    def forward(self, thermal, ambient):
        # thermal here is absolute °C (B,1,H,W); normalize for encoders
        thermal_n = ambient_normalize(thermal, ambient)
        prim = self.inverse(thermal_n)
        t_recon = self.forward_model(
            prim["depth"], prim["emissivity"], prim["surface_temp"], prim["reflection"], ambient
        )
        d = maybe_detach(prim["depth"], self.detach_physics_for_pose)
        e = maybe_detach(prim["emissivity"], self.detach_physics_for_pose)
        tt = maybe_detach(prim["surface_temp"], self.detach_physics_for_pose)
        packed = torch.cat([thermal_n, d, e, tt / 50.0], dim=1)
        spatial = self.thermo(packed)
        cue = torch.cat(
            [d.mean(dim=(2, 3)), e.mean(dim=(2, 3)), tt.mean(dim=(2, 3)) / 50.0, ambient],
            dim=1,
        )
        spatial = self.film(spatial, cue)
        kp = self.kp(packed)
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

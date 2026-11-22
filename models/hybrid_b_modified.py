"""Modified Hybrid B: keep depth on pose graph + depth KP (same fixes as modified_A)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBNReLU, FiLM, KPFeatureExtractor, PositionRetrievalDecoder, ResidualConv, ThermoSpaceEncoder
from .forward_thermal import ForwardThermal, apply_forward
from .physics_utils import ambient_normalize, maybe_detach


class HybridBModified(nn.Module):
    """Like HybridB, but depth stays attached and a depth KP drives joints."""

    has_physics = True

    def __init__(
        self,
        in_ch=1,
        spatial_ch=128,
        kp_dim=96,
        detach_physics_for_pose=True,
        use_forward=True,
        use_residual=True,
        keep_depth_attached=True,
        soft_depth_attach=False,
        use_mask_inv_skip=False,
        depth_attach_alpha=None,
        recon_detach_primitives=False,
        **_kwargs,
    ):
        super().__init__()
        self.detach_physics_for_pose = bool(detach_physics_for_pose)
        self.use_forward = bool(use_forward)
        self.keep_depth_attached = bool(keep_depth_attached)
        self.soft_depth_attach = bool(soft_depth_attach)
        self.use_mask_inv_skip = bool(use_mask_inv_skip)
        self.depth_attach_alpha = None if depth_attach_alpha is None else float(depth_attach_alpha)
        self.recon_detach_primitives = bool(recon_detach_primitives)

        self.thermo = ThermoSpaceEncoder(in_ch, spatial_ch)
        self.prim_head = nn.Sequential(
            ResidualConv(spatial_ch),
            ConvBNReLU(spatial_ch, spatial_ch),
            nn.Conv2d(spatial_ch, 5, 1),
        )
        self.depth_scale = nn.Parameter(torch.tensor(0.6))
        self.depth_bias = nn.Parameter(torch.tensor(0.25))
        self.forward_model = (
            ForwardThermal(use_residual=bool(use_residual)) if self.use_forward else None
        )
        self.film = FiLM(cue_dim=4, ch=spatial_ch)
        # kp: thermal + depth + eps (+ optional mask)
        kp_in = 4 if self.use_mask_inv_skip else 3
        self.kp = KPFeatureExtractor(in_ch=kp_in, feat_dim=kp_dim)
        self.kp_depth = KPFeatureExtractor(in_ch=2, feat_dim=kp_dim)
        self.kp_fuse = nn.Sequential(
            nn.Linear(kp_dim * 2, kp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(kp_dim, kp_dim),
        )
        self.decoder = PositionRetrievalDecoder(spatial_ch, kp_dim)

    def _pose_depth(self, depth: torch.Tensor) -> torch.Tensor:
        if self.keep_depth_attached:
            alpha = self.depth_attach_alpha
            if alpha is not None and self.detach_physics_for_pose:
                a = max(0.0, min(1.0, float(alpha)))
                return a * depth + (1.0 - a) * depth.detach()
            if self.soft_depth_attach and self.detach_physics_for_pose:
                return 0.5 * depth + 0.5 * depth.detach()
            return depth
        return maybe_detach(depth, self.detach_physics_for_pose)

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
        fwd = apply_forward(
            self.forward_model,
            depth,
            eps,
            tt,
            refl,
            ambient,
            detach_primitives=self.recon_detach_primitives,
        )
        d = self._pose_depth(depth)
        e = maybe_detach(eps, self.detach_physics_for_pose)
        cue = torch.cat(
            [d.mean(dim=(2, 3)), e.mean(dim=(2, 3)), tt.mean(dim=(2, 3)) / 50.0, ambient],
            dim=1,
        )
        spatial = self.film(spatial0, cue)
        if self.use_mask_inv_skip:
            m = maybe_detach(torch.sigmoid(mask_logits), self.detach_physics_for_pose)
            kp_in = torch.cat([thermal_n, d, e, m], dim=1)
        else:
            kp_in = torch.cat([thermal_n, d, e], dim=1)
        kp_pack = self.kp(kp_in)
        kp_d = self.kp_depth(torch.cat([thermal_n, d], dim=1))
        kp = self.kp_fuse(torch.cat([kp_pack, kp_d], dim=-1))
        joints = self.decoder(spatial, kp)
        return {
            "joints": joints,
            "depth": depth,
            "t_recon": fwd["t_recon"],
            "t_phys": fwd["t_phys"],
            "t_residual": fwd["t_residual"],
            "mask_logits": mask_logits,
            "emissivity": eps,
            "surface_temp": tt,
            "reflection": refl,
        }

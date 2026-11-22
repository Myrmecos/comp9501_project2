"""Modified Early (A): spatial depth→pose + selective detach so Forward/physics can help joints.

Escalation (plan): mask channel + inverse-feat skip (like HybridD) when use_mask_inv_skip=True.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import FiLM, InverseUNet, KPFeatureExtractor, PositionRetrievalDecoder, ThermoSpaceEncoder
from .forward_thermal import ForwardThermal, apply_forward
from .physics_utils import ambient_normalize, maybe_detach


class HybridAModified(nn.Module):
    """Like HybridA, but depth stays on the pose graph and a depth KP drives joints."""

    has_physics = True

    def __init__(
        self,
        in_ch=1,
        spatial_ch=128,
        kp_dim=96,
        inverse_base=64,
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
        # Fraction of depth gradient that reaches pose when soft-mixing (None → 0.5 if soft else 1.0)
        self.depth_attach_alpha = None if depth_attach_alpha is None else float(depth_attach_alpha)
        # If True, Forward sees stop-grad primitives so recon cannot corrupt inverse/pose
        self.recon_detach_primitives = bool(recon_detach_primitives)
        self.inverse = InverseUNet(in_ch=in_ch, base=inverse_base, out_ch=5)
        self.forward_model = (
            ForwardThermal(use_residual=bool(use_residual)) if self.use_forward else None
        )
        thermo_in = 5 if self.use_mask_inv_skip else 4
        self.thermo = ThermoSpaceEncoder(in_ch=thermo_in, out_ch=spatial_ch)
        self.film = FiLM(cue_dim=4, ch=spatial_ch)
        self.kp = KPFeatureExtractor(in_ch=thermo_in, feat_dim=kp_dim)
        self.kp_depth = KPFeatureExtractor(in_ch=2, feat_dim=kp_dim)
        self.kp_fuse = nn.Sequential(
            nn.Linear(kp_dim * 2, kp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(kp_dim, kp_dim),
        )
        self.decoder = PositionRetrievalDecoder(spatial_ch, kp_dim)
        self.inv_proj = nn.Conv2d(inverse_base, spatial_ch, 1) if self.use_mask_inv_skip else None

    def _pose_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """Depth for pose path: never fully detached when keep_depth_attached."""
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
        prim = self.inverse(thermal_n)
        fwd = apply_forward(
            self.forward_model,
            prim["depth"],
            prim["emissivity"],
            prim["surface_temp"],
            prim["reflection"],
            ambient,
            detach_primitives=self.recon_detach_primitives,
        )
        # Selective detach: ε / Tt follow phase flag; depth follows keep_depth_attached
        d = self._pose_depth(prim["depth"])
        e = maybe_detach(prim["emissivity"], self.detach_physics_for_pose)
        tt = maybe_detach(prim["surface_temp"], self.detach_physics_for_pose)
        if self.use_mask_inv_skip:
            mask = torch.sigmoid(prim["mask_logits"])
            m = maybe_detach(mask, self.detach_physics_for_pose)
            packed = torch.cat([thermal_n, d, e, tt / 50.0, m], dim=1)
        else:
            packed = torch.cat([thermal_n, d, e, tt / 50.0], dim=1)
        spatial = self.thermo(packed)
        if self.inv_proj is not None:
            inv_feat = maybe_detach(prim["feat"], self.detach_physics_for_pose)
            if inv_feat.shape[-2:] != spatial.shape[-2:]:
                inv_feat = F.interpolate(inv_feat, size=spatial.shape[-2:], mode="bilinear", align_corners=False)
            spatial = spatial + self.inv_proj(inv_feat)
        cue = torch.cat(
            [d.mean(dim=(2, 3)), e.mean(dim=(2, 3)), tt.mean(dim=(2, 3)) / 50.0, ambient],
            dim=1,
        )
        spatial = self.film(spatial, cue)
        kp_pack = self.kp(packed)
        kp_d = self.kp_depth(torch.cat([thermal_n, d], dim=1))
        kp = self.kp_fuse(torch.cat([kp_pack, kp_d], dim=-1))
        joints = self.decoder(spatial, kp)
        return {
            "joints": joints,
            "depth": prim["depth"],
            "t_recon": fwd["t_recon"],
            "t_phys": fwd["t_phys"],
            "t_residual": fwd["t_residual"],
            "mask_logits": prim["mask_logits"],
            "emissivity": prim["emissivity"],
            "surface_temp": prim["surface_temp"],
            "reflection": prim["reflection"],
        }

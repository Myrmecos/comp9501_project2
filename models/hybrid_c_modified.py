"""Modified Hybrid C: keep depth attached into late fusion + depth KP on TAPOR path."""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import InverseUNet, KPFeatureExtractor, PositionRetrievalDecoder, ThermoSpaceEncoder
from .forward_thermal import ForwardThermal, apply_forward
from .physics_utils import ambient_normalize, maybe_detach


class HybridCModified(nn.Module):
    """Like HybridC, but depth stays on the fusion graph and conditions the pose path."""

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
        self.depth_attach_alpha = None if depth_attach_alpha is None else float(depth_attach_alpha)
        self.recon_detach_primitives = bool(recon_detach_primitives)

        self.inverse = InverseUNet(in_ch=in_ch, base=inverse_base, out_ch=5)
        self.forward_model = (
            ForwardThermal(use_residual=bool(use_residual)) if self.use_forward else None
        )
        # Pose trunk can optionally see depth/mask like modified_A
        thermo_in = 3 if self.use_mask_inv_skip else 2  # thermal + depth (+ mask)
        self.thermo = ThermoSpaceEncoder(in_ch=thermo_in, out_ch=spatial_ch)
        self.kp = KPFeatureExtractor(in_ch=thermo_in, feat_dim=kp_dim)
        self.kp_depth = KPFeatureExtractor(in_ch=2, feat_dim=kp_dim)
        self.kp_fuse = nn.Sequential(
            nn.Linear(kp_dim * 2, kp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(kp_dim, kp_dim),
        )
        self.decoder = PositionRetrievalDecoder(spatial_ch, kp_dim)
        self.fuse = nn.Sequential(
            nn.Linear(21 * 3 + 6, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 21 * 3 + 1),
        )

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
        mask = torch.sigmoid(prim["mask_logits"])
        d = self._pose_depth(prim["depth"])
        e = maybe_detach(prim["emissivity"], self.detach_physics_for_pose)
        tt = maybe_detach(prim["surface_temp"], self.detach_physics_for_pose)
        m = maybe_detach(mask, self.detach_physics_for_pose)

        if self.use_mask_inv_skip:
            pose_in = torch.cat([thermal_n, d, m], dim=1)
        else:
            pose_in = torch.cat([thermal_n, d], dim=1)
        spatial = self.thermo(pose_in)
        kp_pack = self.kp(pose_in)
        kp_d = self.kp_depth(torch.cat([thermal_n, d], dim=1))
        kp = self.kp_fuse(torch.cat([kp_pack, kp_d], dim=-1))
        joints0 = self.decoder(spatial, kp)

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
            "t_recon": fwd["t_recon"],
            "t_phys": fwd["t_phys"],
            "t_residual": fwd["t_residual"],
            "mask_logits": prim["mask_logits"],
            "emissivity": prim["emissivity"],
            "surface_temp": prim["surface_temp"],
            "reflection": prim["reflection"],
        }

"""Differentiable thermal forward model (TAP3D-inspired, hand-scale)."""
from __future__ import annotations

import torch
import torch.nn as nn


class ForwardThermal(nn.Module):
    """Map primitives → reconstructed temperature (°C).

    Physics branch (Stefan–Boltzmann + reflection + depth attenuation), optionally
    plus a small residual CNN corrector for sensor bias / nonlinearities.

    Set ``use_residual=False`` to force a pure-physics reconstruction (hardens the
    physics constraint: recon loss must be explained by primitives + scalars).
    """

    def __init__(self, residual_ch: int = 16, use_residual: bool = True):
        super().__init__()
        self.use_residual = bool(use_residual)
        self.sigma = nn.Parameter(torch.tensor(5.67e-8, dtype=torch.float32), requires_grad=False)
        self.depth_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.offset = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.residual = nn.Sequential(
            nn.Conv2d(5, residual_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(residual_ch, residual_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(residual_ch, 1, 1),
        )

    def forward_parts(
        self,
        depth: torch.Tensor,
        emissivity: torch.Tensor,
        surface_temp: torch.Tensor,
        reflection: torch.Tensor,
        ambient: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # depth, eps, Tt, R: B,1,H,W ; ambient: B,1
        eps = emissivity.clamp(0.05, 0.99)
        tt = surface_temp
        tt_k = torch.where(tt < 200.0, tt + 273.15, tt)
        amb = ambient.view(-1, 1, 1, 1)
        amb_k = torch.where(amb < 200.0, amb + 273.15, amb)
        emit = eps * self.sigma * (tt_k ** 4)
        refl = (1.0 - eps) * reflection.clamp(0.0, 2.0) * self.sigma * (amb_k ** 4)
        radiance = emit + refl
        d = depth.clamp(0.05, 2.0)
        atten = 1.0 / (1.0 + self.depth_scale.abs() * (d - 0.2).clamp(min=0.0))
        radiance = radiance * atten
        t_phys = self.gain * (radiance / (self.sigma + 1e-12)).clamp(min=1e-6).pow(0.25) - 273.15 + self.offset
        amb_map = amb.expand_as(depth)
        packed = torch.cat([depth, eps, surface_temp / 50.0, reflection, amb_map / 50.0], dim=1)
        if self.use_residual:
            delta = self.residual(packed)
        else:
            delta = t_phys.new_zeros(t_phys.shape)
        t_c = t_phys + delta
        return {"t_phys": t_phys, "t_residual": delta, "t_recon": t_c}

    def forward(
        self,
        depth: torch.Tensor,
        emissivity: torch.Tensor,
        surface_temp: torch.Tensor,
        reflection: torch.Tensor,
        ambient: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_parts(depth, emissivity, surface_temp, reflection, ambient)["t_recon"]


def apply_forward(
    fwd: ForwardThermal | None,
    depth: torch.Tensor,
    emissivity: torch.Tensor,
    surface_temp: torch.Tensor,
    reflection: torch.Tensor,
    ambient: torch.Tensor,
    *,
    detach_primitives: bool = False,
) -> dict[str, torch.Tensor | None]:
    """Run ForwardThermal and return t_recon / t_phys / t_residual (or Nones)."""
    empty = {"t_recon": None, "t_phys": None, "t_residual": None}
    if fwd is None:
        return empty
    if detach_primitives:
        depth = depth.detach()
        emissivity = emissivity.detach()
        surface_temp = surface_temp.detach()
        reflection = reflection.detach()
    return fwd.forward_parts(depth, emissivity, surface_temp, reflection, ambient)

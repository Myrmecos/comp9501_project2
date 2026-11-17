"""Helpers shared by physics hybrids."""
from __future__ import annotations

import torch


def maybe_detach(x: torch.Tensor, detach: bool) -> torch.Tensor:
    return x.detach() if detach else x


def ambient_normalize(thermal: torch.Tensor, ambient: torch.Tensor) -> torch.Tensor:
    """thermal (B,C,H,W) or (B,T,H,W); ambient (B,1)."""
    if thermal.dim() == 4 and thermal.shape[1] <= 4:
        # (B,C,H,W)
        return thermal - ambient.view(-1, 1, 1, 1)
    if thermal.dim() == 4:
        # (B,T,H,W) temporal
        return thermal - ambient.view(-1, 1, 1, 1)
    if thermal.dim() == 3:
        return thermal - ambient.view(-1, 1, 1)
    return thermal

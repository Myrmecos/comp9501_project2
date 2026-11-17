"""Evaluation metrics."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> float:
    # B,21,3 in meters typically → report cm
    err = torch.norm(pred - gt, dim=-1).mean()
    return float(err.detach()) * 100.0


def depth_mae(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        w = mask
        if w.sum() < 1:
            return 0.0
        err = ((pred - gt).abs() * w).sum() / w.sum()
    else:
        err = (pred - gt).abs().mean()
    return float(err.detach()) * 100.0  # cm


def hand_depth_mae(pred_depth: torch.Tensor, hand_depth: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        w = mask
        denom = w.sum(dim=(2, 3)).clamp(min=1.0)
        pred_mean = (pred_depth * w).sum(dim=(2, 3)) / denom
    else:
        pred_mean = pred_depth.mean(dim=(2, 3))
    err = (pred_mean - hand_depth).abs().mean()
    return float(err.detach()) * 100.0


def recon_mae(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is not None and mask.sum() >= 1:
        err = ((pred - gt).abs() * mask).sum() / mask.sum()
    else:
        err = (pred - gt).abs().mean()
    return float(err.detach())


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sample mean over spatial dims; x/mask: B,1,H,W → B,1,1,1."""
    w = mask
    denom = w.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
    return (x * w).sum(dim=(2, 3), keepdim=True) / denom


def shift_invariant_depth_mae(
    pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None
) -> float:
    """Hand depth *shape* MAE (cm), invariant to a global depth offset.

    Removes the masked mean from pred and gt independently, then takes MAE
    under the mask. Moving the whole hand closer/farther (constant shift) does
    not change the score; warping only some pixels does.
    """
    if mask is None:
        mask = torch.ones_like(pred)
    if mask.sum() < 1:
        return 0.0
    pred_c = pred - _masked_mean(pred, mask)
    gt_c = gt - _masked_mean(gt, mask)
    err = ((pred_c - gt_c).abs() * mask).sum() / mask.sum().clamp(min=1.0)
    return float(err.detach()) * 100.0


def depth_gradient_mae(
    pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None
) -> float:
    """MAE of spatial depth gradients (cm / pixel), shift-invariant by construction.

    Compares finite-difference ∂z/∂x and ∂z/∂y under a (shrunk) hand mask.
    Sensitive to local shape distortion; insensitive to global vertical shift.
    """
    # central-ish forward diffs
    dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_g = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_g = gt[:, :, 1:, :] - gt[:, :, :-1, :]

    if mask is None:
        mx = torch.ones_like(dx_p)
        my = torch.ones_like(dy_p)
    else:
        # valid only if both pixels in the pair are hand
        mx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        my = mask[:, :, 1:, :] * mask[:, :, :-1, :]

    n = mx.sum() + my.sum()
    if n < 1:
        return 0.0
    err = ((dx_p - dx_g).abs() * mx).sum() + ((dy_p - dy_g).abs() * my).sum()
    err = err / n.clamp(min=1.0)
    return float(err.detach()) * 100.0


def depth_shape_corr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Pearson correlation of masked depth values (higher better; ∈ [-1, 1]).

    Invariant to global affine a·z+b (scale + shift). Complements SI-MAE.
    """
    if mask is None:
        mask = torch.ones_like(pred)
    # flatten per batch then average correlations
    b = pred.shape[0]
    corrs = []
    for i in range(b):
        m = mask[i] > 0.5
        if m.sum() < 2:
            continue
        p = pred[i][m]
        g = gt[i][m]
        p = p - p.mean()
        g = g - g.mean()
        denom = p.norm() * g.norm()
        if float(denom) < 1e-12:
            continue
        corrs.append(float((p * g).sum() / denom))
    if not corrs:
        return float("nan")
    return float(sum(corrs) / len(corrs))

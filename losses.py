"""Training losses for full TAPOR baseline and physics hybrids."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from utils import HAND_BONES


def joint_mse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, gt)


def bone_length_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    loss = pred.new_tensor(0.0)
    for a, b in HAND_BONES:
        pred_len = torch.norm(pred[:, a] - pred[:, b], dim=-1)
        gt_len = torch.norm(gt[:, a] - gt[:, b], dim=-1)
        loss = loss + F.mse_loss(pred_len, gt_len)
    return loss / len(HAND_BONES)


def huber_depth(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        w = mask
        if w.sum() < 1:
            return pred.new_tensor(0.0)
        return F.smooth_l1_loss(pred * w, gt * w, reduction="sum") / w.sum().clamp(min=1.0)
    return F.smooth_l1_loss(pred, gt)


def hand_depth_loss(
    pred_depth: torch.Tensor, hand_depth: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Align masked mean predicted depth with scalar hand_depth (meters)."""
    if mask is not None:
        w = mask
        denom = w.sum(dim=(2, 3)).clamp(min=1.0)
        pred_mean = (pred_depth * w).sum(dim=(2, 3)) / denom
    else:
        pred_mean = pred_depth.mean(dim=(2, 3))
    return F.smooth_l1_loss(pred_mean, hand_depth)


def recon_loss(pred_t: torch.Tensor, gt_t: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    return huber_depth(pred_t, gt_t, mask)


def mask_bce(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(pred, gt)


def resolve_loss_weights(cfg: dict, epoch: int) -> dict:
    """Phase-1 emphasizes hand physics; phase-2 emphasizes joints."""
    out = dict(cfg)
    p1 = int(cfg.get("phase1_epochs", 0))
    if p1 > 0 and epoch <= p1:
        out["lambda_joint"] = float(cfg.get("phase1_lambda_joint", cfg.get("lambda_joint", 1.0)))
        out["lambda_bone"] = float(cfg.get("phase1_lambda_bone", cfg.get("lambda_bone", 1.0)))
        out["lambda_depth"] = float(cfg.get("phase1_lambda_depth", cfg.get("lambda_depth", 0.1)))
        out["lambda_hand_depth"] = float(
            cfg.get("phase1_lambda_hand_depth", cfg.get("lambda_hand_depth", 2.0))
        )
        out["lambda_recon"] = float(cfg.get("phase1_lambda_recon", cfg.get("lambda_recon", 1.0)))
        out["lambda_mask"] = float(cfg.get("phase1_lambda_mask", cfg.get("lambda_mask", 0.2)))
        out["detach_physics_for_pose"] = bool(cfg.get("phase1_detach_physics", False))
    else:
        out["lambda_joint"] = float(cfg.get("phase2_lambda_joint", cfg.get("lambda_joint", 1.0)))
        out["lambda_bone"] = float(cfg.get("phase2_lambda_bone", cfg.get("lambda_bone", 1.0)))
        out["lambda_depth"] = float(cfg.get("phase2_lambda_depth", cfg.get("lambda_depth", 0.1)))
        out["lambda_hand_depth"] = float(
            cfg.get("phase2_lambda_hand_depth", cfg.get("lambda_hand_depth", 1.0))
        )
        out["lambda_recon"] = float(cfg.get("phase2_lambda_recon", cfg.get("lambda_recon", 0.5)))
        out["lambda_mask"] = float(cfg.get("phase2_lambda_mask", cfg.get("lambda_mask", 0.1)))
        out["detach_physics_for_pose"] = bool(
            cfg.get("phase2_detach_physics", cfg.get("detach_physics_for_pose", True))
        )
    return out


def total_loss(outputs: dict, batch: dict, cfg: dict, has_physics: bool) -> tuple[torch.Tensor, dict]:
    parts = {}
    loss = outputs["joints"].new_tensor(0.0)
    lj = joint_mse(outputs["joints"], batch["joints"])
    lb = bone_length_loss(outputs["joints"], batch["joints"])
    parts["joint"] = float(lj.detach())
    parts["bone"] = float(lb.detach())
    loss = loss + cfg.get("lambda_joint", 1.0) * lj + cfg.get("lambda_bone", 1.0) * lb

    if has_physics:
        mask = batch.get("mask")
        if "depth" in outputs and outputs["depth"] is not None:
            # Hand-focused map depth (zeros off-hand in GT); supervise under mask only.
            ld = huber_depth(outputs["depth"], batch["depth"], mask)
            parts["depth"] = float(ld.detach())
            loss = loss + cfg.get("lambda_depth", 0.1) * ld
            if "hand_depth" in batch:
                lh = hand_depth_loss(outputs["depth"], batch["hand_depth"], mask)
                parts["hand_depth"] = float(lh.detach())
                loss = loss + cfg.get("lambda_hand_depth", 1.0) * lh
        if "t_recon" in outputs and outputs["t_recon"] is not None:
            # Reconstruction focused on hand pixels only (not background).
            lr = recon_loss(outputs["t_recon"], batch["thermal_abs"], mask)
            parts["recon"] = float(lr.detach())
            loss = loss + cfg.get("lambda_recon", 0.5) * lr
        if "mask_logits" in outputs and outputs["mask_logits"] is not None and mask is not None:
            lm = mask_bce(outputs["mask_logits"], mask)
            parts["mask"] = float(lm.detach())
            loss = loss + cfg.get("lambda_mask", 0.1) * lm

    # Hand3d heatmap supervision (TAPOR baseline3d)
    if outputs.get("heatmap") is not None and batch.get("heatmap") is not None:
        lhmap = F.mse_loss(outputs["heatmap"], batch["heatmap"])
        parts["heatmap"] = float(lhmap.detach())
        loss = loss + cfg.get("lambda_heatmap", 1.0) * lhmap

    parts["total"] = float(loss.detach())
    return loss, parts

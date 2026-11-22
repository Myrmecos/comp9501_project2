#!/usr/bin/env python3
"""Unified training for TAPOR baselines and physics hybrids (small or full split)."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from losses import resolve_loss_weights, total_loss
from metrics import depth_mae, hand_depth_mae, mpjpe, recon_mae
from models import build_model
from utils import TaporSmallDataset, collate_batch, load_yaml

MODEL_CHOICES = [
    "vanilla",
    "A",
    "modified_A",
    "B",
    "modified_B",
    "C",
    "modified_C",
    "D",
    "modified_D",
    "mediapipe",
    "nanotapor",
    "hand3d",
    "resnet",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_batch(batch: dict, device: torch.device) -> dict:
    batch = {k: v.to(device=device, dtype=torch.float32) for k, v in batch.items()}
    th = batch["thermal"]
    if th.dim() == 4 and th.shape[1] == 1:
        batch["thermal_abs"] = th
    elif th.dim() == 4:
        batch["thermal_abs"] = th[:, -1:, :, :]
    else:
        batch["thermal_abs"] = th.unsqueeze(1)
    return batch


def forward_model(model, batch):
    kwargs = {}
    if getattr(model, "needs_hand_side", False):
        kwargs["hand_side"] = batch["hand_side"]
    return model(batch["thermal"], batch["ambient"], **kwargs) if kwargs else model(
        batch["thermal"], batch["ambient"]
    )


def evaluate(model, loader, device, has_physics: bool):
    model.eval()
    mpjpes, depths, hand_ds, recons = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = prepare_batch(batch, device)
            out = forward_model(model, batch)
            mpjpes.append(mpjpe(out["joints"], batch["joints"]))
            if has_physics and out.get("depth") is not None:
                depths.append(depth_mae(out["depth"], batch["depth"], batch["mask"]))
                hand_ds.append(hand_depth_mae(out["depth"], batch["hand_depth"], batch["mask"]))
            if has_physics and out.get("t_recon") is not None:
                recons.append(recon_mae(out["t_recon"], batch["thermal_abs"], batch["mask"]))
    return {
        "mpjpe_cm": float(np.mean(mpjpes)) if mpjpes else float("nan"),
        "depth_mae_cm": float(np.mean(depths)) if depths else float("nan"),
        "hand_depth_mae_cm": float(np.mean(hand_ds)) if hand_ds else float("nan"),
        "recon_mae": float(np.mean(recons)) if recons else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES)
    parser.add_argument("--exp-id", type=str, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/small_split.yml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/tapor"))
    parser.add_argument("--split-json", type=Path, default=Path("data/tapor/small_split.json"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint: 'last', 'best', or a path to a .pt file",
    )
    parser.add_argument(
        "--load-weights",
        type=str,
        default=None,
        help="Load state_dict only from a .pt path and start from epoch 1 (finetune from SSL)",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="If set, write weights/ and log/ under this root (e.g. small_ws) instead of repo root",
    )
    args = parser.parse_args()
    if args.resume and args.load_weights:
        raise SystemExit("Use only one of --resume or --load-weights")

    cfg = load_yaml(args.config)
    overrides = (cfg.get("model_overrides") or {}).get(args.model, {})
    for k, v in overrides.items():
        cfg[k] = v

    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.max_minutes is not None:
        cfg["max_minutes"] = args.max_minutes
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    if args.lr is not None:
        cfg["lr"] = args.lr

    set_seed(int(cfg.get("seed", 0)))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = int(cfg["batch_size"])

    num_history = int(cfg.get("num_history", 0)) if args.model == "vanilla" else 0
    require_heatmap = args.model == "hand3d"
    # full / noleak / limited_label / micro splits already enumerate frames — do not subsample
    sj = str(args.split_json)
    max_frames = None if (
        "full_split" in sj
        or "noleak" in sj
        or "limited_label" in sj
        or "micro_split" in sj
    ) else 80

    art = Path(args.artifact_root) if args.artifact_root else Path(".")
    log_dir = art / "log" / "tb" / args.exp_id
    weight_dir = art / "weights" / args.exp_id
    log_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)
    (art / "log").mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))

    ds_kwargs = dict(
        data_root=args.data_root,
        split_json=args.split_json,
        height=int(cfg.get("height", 24)),
        width=int(cfg.get("width", 32)),
        num_history=num_history,
        mask_top_frac=float(cfg.get("mask_top_frac", 0.25)),
        mask_min_delta=float(cfg.get("mask_min_delta", 6.0)),
        require_heatmap=require_heatmap,
        heatmap_size=256,
        max_frames_per_file=max_frames,
    )
    train_ds = TaporSmallDataset(split="train", **ds_kwargs)
    val_ds = TaporSmallDataset(split="val", **ds_kwargs)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 2)),
        collate_fn=collate_batch,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 2)),
        collate_fn=collate_batch,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(
        args.model,
        batch_size=batch_size,
        device=device,
        num_history=num_history,
        detach_physics_for_pose=bool(cfg.get("detach_physics_for_pose", True)),
        spatial_ch=int(cfg.get("spatial_ch", 128)),
        kp_dim=int(cfg.get("kp_dim", 96)),
        inverse_base=int(cfg.get("inverse_base", 64)),
        use_forward=bool(cfg.get("use_forward", True)),
        keep_depth_attached=bool(cfg.get("keep_depth_attached", True)),
        soft_depth_attach=bool(cfg.get("soft_depth_attach", False)),
        use_mask_inv_skip=bool(cfg.get("use_mask_inv_skip", False)),
        depth_attach_alpha=(
            None if cfg.get("depth_attach_alpha") is None else float(cfg.get("depth_attach_alpha"))
        ),
        recon_detach_primitives=bool(cfg.get("recon_detach_primitives", False)),
        use_residual=bool(cfg.get("use_residual", True)),
    ).float().to(device)
    has_physics = bool(
        getattr(
            model,
            "has_physics",
            args.model not in ("vanilla", "mediapipe", "nanotapor", "hand3d", "resnet"),
        )
    )

    opt_name = str(cfg.get("optimizer", "adamw")).lower()
    lr = float(cfg["lr"])
    wd = float(cfg.get("weight_decay", 1e-5))
    if opt_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    early_stop_metric = str(cfg.get("early_stop_metric", "mpjpe_cm"))
    best_score = float("inf")
    best_metrics = {}
    patience = int(cfg.get("early_stop_patience", 8))
    bad = 0
    start_epoch = 1
    prior_elapsed = 0.0
    t0 = time.time()
    max_seconds = float(cfg.get("max_minutes", 55)) * 60.0
    history = []

    if args.load_weights:
        load_path = Path(args.load_weights)
        if not load_path.exists():
            raise FileNotFoundError(f"Load-weights checkpoint not found: {load_path}")
        ckpt = torch.load(load_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        start_epoch = 1
        print(
            f"[{args.exp_id}] loaded weights from {load_path} "
            f"(src_epoch={ckpt.get('epoch')}, src_exp={ckpt.get('exp_id')}); start_epoch=1",
            flush=True,
        )

    if bool(cfg.get("freeze_physics", False)):
        frozen = []
        for name in ("inverse", "forward_model"):
            mod = getattr(model, name, None)
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad_(False)
            frozen.append(name)
        # rebuild optimizer on trainable params only
        trainable = [p for p in model.parameters() if p.requires_grad]
        if opt_name == "adamw":
            opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
        else:
            opt = torch.optim.Adam(trainable, lr=lr)
        print(f"[{args.exp_id}] freeze_physics={frozen} trainable_tensors={len(trainable)}", flush=True)

    if args.resume:
        if args.resume in ("last", "best"):
            resume_path = weight_dir / f"{args.resume}.pt"
        else:
            resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            try:
                opt.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[{args.exp_id}] warn: could not load optimizer state ({e})", flush=True)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        # Restore best from best.pt when resuming from last
        best_path = weight_dir / "best.pt"
        if best_path.exists():
            best_ck = torch.load(best_path, map_location="cpu", weights_only=False)
            best_metrics = dict(best_ck.get("val_metrics") or {})
            if "epoch" in best_ck:
                best_metrics["epoch"] = best_ck["epoch"]
            best_score = float(best_metrics.get(early_stop_metric, best_metrics.get("mpjpe_cm", float("inf"))))
            bad = max(0, int(ckpt.get("epoch", 0)) - int(best_ck.get("epoch", 0)))
        else:
            best_metrics = dict(ckpt.get("val_metrics") or {})
            best_metrics["epoch"] = ckpt.get("epoch")
            best_score = float(best_metrics.get(early_stop_metric, best_metrics.get("mpjpe_cm", float("inf"))))
        prior_summary = weight_dir / "summary.json"
        if prior_summary.exists():
            try:
                prior_elapsed = float(json.loads(prior_summary.read_text()).get("elapsed_sec", 0.0))
            except Exception:
                prior_elapsed = 0.0
        print(
            f"[{args.exp_id}] resumed from {resume_path} at epoch {start_epoch} "
            f"(best_{early_stop_metric}={best_score:.3f} @ ep {best_metrics.get('epoch')}, bad={bad})",
            flush=True,
        )

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"[{args.exp_id}] model={args.model} params={n_params:.3f}M device={device} "
        f"bs={batch_size} lr={lr} num_history={num_history} "
        f"train={len(train_ds)} val={len(val_ds)} start_epoch={start_epoch}",
        flush=True,
    )

    for epoch in range(start_epoch, int(cfg["max_epochs"]) + 1):
        if time.time() - t0 > max_seconds:
            print(f"[{args.exp_id}] hit max_minutes={cfg['max_minutes']}, stopping.", flush=True)
            break

        epoch_cfg = resolve_loss_weights(cfg, epoch)
        if has_physics and hasattr(model, "detach_physics_for_pose"):
            model.detach_physics_for_pose = bool(epoch_cfg.get("detach_physics_for_pose", True))

        model.train()
        running = []
        for batch in train_loader:
            if time.time() - t0 > max_seconds:
                break
            batch = prepare_batch(batch, device)
            out = forward_model(model, batch)
            loss, parts = total_loss(out, batch, epoch_cfg, has_physics)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running.append(parts)

        def avg_key(key):
            vals = [r[key] for r in running if key in r]
            return float(np.mean(vals)) if vals else float("nan")

        train_keys = (
            "total",
            "joint",
            "bone",
            "depth",
            "hand_depth",
            "joint_depth",
            "recon",
            "recon_phys",
            "residual",
            "mask",
            "heatmap",
        )
        train_metrics = {f"train/{k}": avg_key(k) for k in train_keys}
        val_metrics = evaluate(model, val_loader, device, has_physics)
        for k, v in train_metrics.items():
            if np.isfinite(v):
                writer.add_scalar(k, v, epoch)
        writer.add_scalar("val/mpjpe_cm", val_metrics["mpjpe_cm"], epoch)
        if np.isfinite(val_metrics["depth_mae_cm"]):
            writer.add_scalar("val/depth_mae_cm", val_metrics["depth_mae_cm"], epoch)
        if np.isfinite(val_metrics["hand_depth_mae_cm"]):
            writer.add_scalar("val/hand_depth_mae_cm", val_metrics["hand_depth_mae_cm"], epoch)
        if np.isfinite(val_metrics["recon_mae"]):
            writer.add_scalar("val/recon_mae", val_metrics["recon_mae"], epoch)

        row = {
            "epoch": epoch,
            **{k.replace("train/", "train_"): v for k, v in train_metrics.items()},
            **val_metrics,
        }
        history.append(row)
        print(
            f"[{args.exp_id}] epoch={epoch} train_loss={train_metrics['train/total']:.4f} "
            f"val_mpjpe_cm={val_metrics['mpjpe_cm']:.3f} "
            f"hand_depth_cm={val_metrics['hand_depth_mae_cm']:.3f} "
            f"depth_cm={val_metrics['depth_mae_cm']:.3f} recon={val_metrics['recon_mae']:.3f}",
            flush=True,
        )

        ckpt = {
            "epoch": epoch,
            "model": args.model,
            "exp_id": args.exp_id,
            "state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
            "val_metrics": val_metrics,
            "cfg": cfg,
            "num_history": num_history,
        }
        torch.save(ckpt, weight_dir / "last.pt")
        score = float(val_metrics.get(early_stop_metric, float("nan")))
        if not np.isfinite(score):
            # Fallback if metric missing (e.g. recon on nofwd)
            score = float(val_metrics.get("mpjpe_cm", float("inf")))
        if score < best_score:
            best_score = score
            best_metrics = dict(val_metrics)
            best_metrics["epoch"] = epoch
            best_metrics["early_stop_metric"] = early_stop_metric
            torch.save(ckpt, weight_dir / "best.pt")
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[{args.exp_id}] early stop at epoch {epoch}", flush=True)
                break

    elapsed = time.time() - t0 + prior_elapsed
    summary = {
        "exp_id": args.exp_id,
        "model": args.model,
        "elapsed_sec": elapsed,
        "best": best_metrics,
        "early_stop_metric": early_stop_metric,
        "train_frames": len(train_ds),
        "val_frames": len(val_ds),
        "params_m": n_params,
        "num_history": num_history,
        "resumed_from_epoch": start_epoch - 1 if args.resume else None,
        "loaded_weights": args.load_weights,
        "history_tail": history[-5:],
    }
    (weight_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (art / "log" / f"{args.exp_id}_summary.json").write_text(json.dumps(summary, indent=2))
    writer.close()
    best_mpjpe = float(best_metrics.get("mpjpe_cm", float("nan"))) if best_metrics else float("nan")
    print(
        f"[{args.exp_id}] done best_{early_stop_metric}={best_score:.3f} "
        f"best_mpjpe_cm={best_mpjpe:.3f} elapsed_s={elapsed:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

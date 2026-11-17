#!/usr/bin/env python3
"""Evaluate trained checkpoints on official TAPOR Testset (inference.py file list)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import forward_model, prepare_batch
from metrics import (
    depth_gradient_mae,
    depth_mae,
    depth_shape_corr,
    hand_depth_mae,
    mpjpe,
    recon_mae,
    shift_invariant_depth_mae,
)
from models import build_model
from utils import TaporSmallDataset, collate_batch

EXPS = [
    ("hand3d", "hand3d_full"),
    ("mediapipe", "mediapipe_full"),
    ("nanotapor", "nanotapor_full"),
    ("vanilla", "vanilla_full"),
    ("A", "hybridA_full"),
    ("B", "hybridB_full"),
    ("C", "hybridC_full"),
    ("D", "hybridD_full"),
]


def _nanmean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def eval_ckpt(model_name: str, ckpt_path: Path, split_json: Path, device: torch.device, batch_size: int = 16):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg") or {}
    num_history = int(ckpt.get("num_history", cfg.get("num_history", 0) if model_name == "vanilla" else 0))
    if model_name != "vanilla":
        num_history = 0
    require_heatmap = model_name == "hand3d"

    ds = TaporSmallDataset(
        data_root=Path("data/tapor"),
        split_json=split_json,
        split="test",
        num_history=num_history,
        max_frames_per_file=None,
        require_heatmap=require_heatmap,
        mask_top_frac=float(cfg.get("mask_top_frac", 0.25)),
        mask_min_delta=float(cfg.get("mask_min_delta", 6.0)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_batch)

    model = build_model(
        model_name,
        batch_size=batch_size,
        device=device,
        num_history=num_history,
        detach_physics_for_pose=True,
        spatial_ch=int(cfg.get("spatial_ch", 128)),
        kp_dim=int(cfg.get("kp_dim", 96)),
        inverse_base=int(cfg.get("inverse_base", 64)),
    ).float().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    has_physics = bool(getattr(model, "has_physics", False))

    mpjpes, depths, hands, recons = [], [], [], []
    si_maes, grad_maes, shape_corrs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = prepare_batch(batch, device)
            # pad last batch for vanilla fixed PE if needed — model handles padding
            out = forward_model(model, batch)
            # per-sample mpjpe
            err = torch.norm(out["joints"] - batch["joints"], dim=-1).mean(dim=-1) * 100.0
            mpjpes.append(err.cpu().numpy())
            if has_physics and out.get("depth") is not None:
                depths.append(depth_mae(out["depth"], batch["depth"], batch["mask"]))
                hands.append(hand_depth_mae(out["depth"], batch["hand_depth"], batch["mask"]))
                si_maes.append(shift_invariant_depth_mae(out["depth"], batch["depth"], batch["mask"]))
                grad_maes.append(depth_gradient_mae(out["depth"], batch["depth"], batch["mask"]))
                shape_corrs.append(depth_shape_corr(out["depth"], batch["depth"], batch["mask"]))
            if has_physics and out.get("t_recon") is not None:
                recons.append(recon_mae(out["t_recon"], batch["thermal_abs"], batch["mask"]))

    mpjpes = np.concatenate(mpjpes)
    return {
        "ckpt": str(ckpt_path),
        "train_best_epoch": (ckpt.get("val_metrics") or {}).get("epoch", ckpt.get("epoch")),
        "train_val_mpjpe_cm": (ckpt.get("val_metrics") or {}).get("mpjpe_cm"),
        "test_frames": int(len(ds)),
        "test_mpjpe_cm": float(mpjpes.mean()),
        "test_mpjpe_median_cm": float(np.median(mpjpes)),
        "test_mpjpe_std_cm": float(mpjpes.std()),
        "test_hand_depth_mae_cm": _nanmean(hands),
        "test_depth_mae_cm": _nanmean(depths),
        "test_recon_mae": _nanmean(recons),
        "test_si_depth_mae_cm": _nanmean(si_maes),
        "test_depth_grad_mae_cm": _nanmean(grad_maes),
        "test_depth_shape_corr": _nanmean(shape_corrs),
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
        "num_history": num_history,
    }


def _fmt(x: float, nd: int = 3) -> str:
    x = float(x)
    return f"{x:.{nd}f}" if np.isfinite(x) else "nan"


def write_markdown(results: list[dict], split_json: Path, out_md: Path, out_json: Path) -> None:
    lines = [
        "# Official TAPOR Testset evaluation",
        "",
        "Test files from `third_party/TAPOR/inference.py` (paper eval list).",
        f"Split: `{split_json}`",
        "",
        "| Model | Exp | Test MPJPE (cm) | Median | Hand-depth MAE (cm) | Depth MAE (cm) | Recon | Train-val MPJPE | Frames | Status |",
        "|-------|-----|-----------------|--------|---------------------|----------------|-------|-----------------|--------|--------|",
    ]
    for r in results:
        if r.get("status") != "ok":
            lines.append(
                f"| {r.get('model')} | {r.get('exp')} | - | - | - | - | - | - | - | {r.get('status')} |"
            )
            continue
        lines.append(
            "| {model} | {exp} | {mpjpe} | {med} | {hand} | {depth} | {recon} | {tr} | {n} | ok |".format(
                model=r["model"],
                exp=r["exp"],
                mpjpe=_fmt(r["test_mpjpe_cm"]),
                med=_fmt(r["test_mpjpe_median_cm"]),
                hand=_fmt(r["test_hand_depth_mae_cm"]),
                depth=_fmt(r["test_depth_mae_cm"]),
                recon=_fmt(r["test_recon_mae"]),
                tr=_fmt(r.get("train_val_mpjpe_cm") or float("nan")),
                n=r["test_frames"],
            )
        )

    lines += [
        "",
        "## Hand depth shape (shift-invariant)",
        "",
        "Metrics that ignore a global hand depth offset (whole hand closer/farther) but "
        "penalize local shape distortion:",
        "",
        "- **SI-MAE** (`shift_invariant_depth_mae`): mean-center pred/gt under the hand mask, then MAE (cm).",
        "- **Grad-MAE** (`depth_gradient_mae`): MAE of spatial depth gradients (cm/px); shift-invariant by construction.",
        "- **Shape corr** (`depth_shape_corr`): Pearson correlation of masked depths (higher better; also scale-invariant).",
        "",
        "Pose-only baselines have no depth head → `nan`.",
        "",
        "| Model | Exp | Depth MAE (cm) | SI-MAE (cm) ↓ | Grad-MAE (cm/px) ↓ | Shape corr ↑ | Status |",
        "|-------|-----|----------------|---------------|--------------------|--------------|--------|",
    ]
    for r in results:
        if r.get("status") != "ok":
            lines.append(
                f"| {r.get('model')} | {r.get('exp')} | - | - | - | - | {r.get('status')} |"
            )
            continue
        lines.append(
            "| {model} | {exp} | {depth} | {si} | {grad} | {corr} | ok |".format(
                model=r["model"],
                exp=r["exp"],
                depth=_fmt(r.get("test_depth_mae_cm", float("nan"))),
                si=_fmt(r.get("test_si_depth_mae_cm", float("nan"))),
                grad=_fmt(r.get("test_depth_grad_mae_cm", float("nan"))),
                corr=_fmt(r.get("test_depth_shape_corr", float("nan"))),
            )
        )

    lines.append("")
    lines.append("Paper reference (full test): Tapor **2.26** cm, MediaPipe3d **3.32**, NanoTapor **3.27**, Hand3d **4.11**.")
    lines.append("")
    lines.append(f"Raw JSON: `{out_json}`")
    out_md.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-json", type=Path, default=Path("data/tapor/official_test_split.json"))
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("log/OFFICIAL_TEST_RESULTS.md"))
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional model names to evaluate (e.g. A B C D). Others kept from existing JSON if --merge.",
    )
    ap.add_argument(
        "--merge",
        action="store_true",
        help="Merge evaluated rows into existing log/official_test_results.json by model name.",
    )
    args = ap.parse_args()

    if not args.split_json.exists():
        raise SystemExit(f"Missing {args.split_json}; run scripts/make_official_test_split.py first")

    out_json = Path("log/official_test_results.json")
    existing = {}
    if args.merge and out_json.exists():
        for row in json.loads(out_json.read_text()):
            existing[row.get("model")] = row

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    only = set(args.only) if args.only else None
    results_by_model: dict[str, dict] = dict(existing) if args.merge else {}

    for model_name, exp in EXPS:
        if only is not None and model_name not in only:
            if model_name not in results_by_model:
                results_by_model[model_name] = {"model": model_name, "exp": exp, "status": "skipped"}
            continue
        ckpt = Path("weights") / exp / "best.pt"
        if not ckpt.exists():
            print(f"[skip] {exp}: no best.pt")
            results_by_model[model_name] = {"model": model_name, "exp": exp, "status": "missing"}
            continue
        print(f"[eval] {exp} ...", flush=True)
        # hand3d heatmaps are memory-heavy; smaller batch
        bs = 8 if model_name == "hand3d" else args.batch_size
        if model_name == "vanilla":
            bs = min(bs, 24)
        try:
            metrics = eval_ckpt(model_name, ckpt, args.split_json, device, batch_size=bs)
            metrics.update({"model": model_name, "exp": exp, "status": "ok"})
            results_by_model[model_name] = metrics
            print(
                f"  test_mpjpe={metrics['test_mpjpe_cm']:.3f} cm "
                f"(median {metrics['test_mpjpe_median_cm']:.3f}) "
                f"hand={metrics['test_hand_depth_mae_cm']:.3f} "
                f"si={metrics['test_si_depth_mae_cm']:.3f} "
                f"grad={metrics['test_depth_grad_mae_cm']:.3f} "
                f"corr={metrics['test_depth_shape_corr']:.3f} "
                f"n={metrics['test_frames']}",
                flush=True,
            )
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            results_by_model[model_name] = {"model": model_name, "exp": exp, "status": f"error: {e}"}
            torch.cuda.empty_cache()

    # preserve EXPS order
    results = [results_by_model.get(m, {"model": m, "exp": e, "status": "missing"}) for m, e in EXPS]
    out_json.write_text(json.dumps(results, indent=2))
    write_markdown(results, args.split_json, args.out, out_json)
    print(args.out.read_text())


if __name__ == "__main__":
    main()

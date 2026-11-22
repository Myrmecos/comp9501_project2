#!/usr/bin/env python3
"""Regenerate presentation figures 09–11.

  09: MPJPE bar (baselines + Early/Trunk/Late/Map; no Hand3d)
  10: depth / hand-depth MAE + thermal recon MAE
  11: parameters (M) vs MPJPE scatter (no Hand3d)

Reads official Testset metrics from log/official_test_results.json.
Hybrid labels: A→Early, B→Trunk, C→Late, D→Map.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# PhyPose naming (matches system_diagram/four_models.md)
HYBRID_LABELS = {
    "hybridA_full": "Early",
    "hybridB_full": "Trunk",
    "hybridC_full": "Late",
    "hybridD_full": "Map",
}

# Paul Tol muted triad for fig 10
C_DEPTH = "#88CCEE"
C_HAND = "#4c78a8"
C_RECON = "#4c78a8"
C_MPJPE = "#4c78a8"


def _fnum(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load_by_exp() -> dict:
    path = ROOT / "log" / "official_test_results.json"
    rows = json.loads(path.read_text())
    return {r["exp"]: r for r in rows if r.get("status") == "ok"}


def style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_09_mpjpe(by_exp: dict) -> Path:
    """Baselines + PhyPose Early/Trunk/Late/Map MPJPE (no Hand3d, no paper line)."""
    order = [
        ("mediapipe_full", "MediaPipe3d"),
        ("nanotapor_full", "NanoTapor"),
        ("vanilla_full", "TAPOR"),
        ("hybridA_full", HYBRID_LABELS["hybridA_full"]),
        ("hybridB_full", HYBRID_LABELS["hybridB_full"]),
        ("hybridC_full", HYBRID_LABELS["hybridC_full"]),
        ("hybridD_full", HYBRID_LABELS["hybridD_full"]),
    ]
    vals = [_fnum(by_exp[e]["test_mpjpe_cm"]) for e, _ in order]
    labels = [lab for _, lab in order]

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=C_MPJPE, width=0.72, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("MPJPE (cm)")
    ax.grid(axis="y", alpha=0.35, linestyle="--", zorder=0)
    ax.set_ylim(0, max(vals) * 1.18)
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.06,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    path = OUT / "09_mpjpe_baselines_hybrids.png"
    fig.savefig(path, dpi=160, facecolor="white", edgecolor="none")
    plt.close(fig)
    return path


def plot_10_depth_thermal(by_exp: dict) -> Path:
    """PhyPose Early/Trunk/Late/Map: depth / hand-depth MAE + thermal recon MAE."""
    hyb = [
        ("hybridA_full", HYBRID_LABELS["hybridA_full"]),
        ("hybridB_full", HYBRID_LABELS["hybridB_full"]),
        ("hybridC_full", HYBRID_LABELS["hybridC_full"]),
        ("hybridD_full", HYBRID_LABELS["hybridD_full"]),
    ]
    depth = [_fnum(by_exp[e]["test_depth_mae_cm"]) for e, _ in hyb]
    hand = [_fnum(by_exp[e]["test_hand_depth_mae_cm"]) for e, _ in hyb]
    recon = [_fnum(by_exp[e]["test_recon_mae"]) for e, _ in hyb]
    labs = [lab for _, lab in hyb]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    xx = np.arange(len(labs))

    ax = axes[0]
    w = 0.36
    b1 = ax.bar(xx - w / 2, depth, width=w, label="Depth MAE", color=C_DEPTH, zorder=3)
    b2 = ax.bar(xx + w / 2, hand, width=w, label="Hand-depth MAE", color=C_HAND, zorder=3)
    ax.set_xticks(xx)
    ax.set_xticklabels(labs)
    ax.set_ylabel("MAE (cm)")
    ax.set_title("Depth Error")
    ax.set_facecolor("white")
    ax.grid(axis="y", alpha=0.28, linestyle="--", zorder=0, color="#bbbbbb")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, max(depth + hand) * 1.2)
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.08,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#222",
            )

    ax = axes[1]
    bars = ax.bar(xx, recon, color=C_RECON, width=0.65, zorder=3)
    ax.set_xticks(xx)
    ax.set_xticklabels(labs)
    ax.set_ylabel("MAE (°C)")
    ax.set_title("Thermal Reconstruction Error")
    ax.set_facecolor("white")
    ax.grid(axis="y", alpha=0.28, linestyle="--", zorder=0, color="#bbbbbb")
    ax.set_ylim(0, max(recon) * 1.25)
    for b, v in zip(bars, recon):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.005,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222",
        )

    fig.patch.set_facecolor("white")
    fig.tight_layout()
    path = OUT / "10_hybrids_depth_thermal_mae.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return path


def plot_11_params_vs_mpjpe(by_exp: dict) -> Path:
    """Params (M) vs official Test MPJPE; baselines + PhyPose, no Hand3d."""
    order = [
        ("mediapipe_full", "MediaPipe3d"),
        ("nanotapor_full", "NanoTapor"),
        ("vanilla_full", "TAPOR"),
        ("hybridA_full", HYBRID_LABELS["hybridA_full"]),
        ("hybridB_full", HYBRID_LABELS["hybridB_full"]),
        ("hybridC_full", HYBRID_LABELS["hybridC_full"]),
        ("hybridD_full", HYBRID_LABELS["hybridD_full"]),
    ]
    xs, ys, labs = [], [], []
    for exp, lab in order:
        r = by_exp[exp]
        xs.append(_fnum(r["params_m"]))
        ys.append(_fnum(r["test_mpjpe_cm"]))
        labs.append(lab)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.scatter(xs, ys, s=90, c=C_MPJPE, zorder=3)
    # Per-label offsets (points): keep Early/Late/Map readable at ~22M
    offsets = {
        "NanoTapor": (8, 6),
        "MediaPipe3d": (8, 6),
        "TAPOR": (8, -12),
        "Trunk": (8, 6),
        "Early": (8, -14),
        "Late": (8, 8),
        "Map": (8, 8),
    }
    for x, y, lab in zip(xs, ys, labs):
        ox, oy = offsets.get(lab, (8, 6))
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(ox, oy), fontsize=10)
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("MPJPE (cm)")
    ax.grid(True, alpha=0.35, linestyle="--", zorder=0)
    ax.set_xlim(-0.5, max(xs) * 1.15)
    lo, hi = min(ys), max(ys)
    ax.set_ylim(lo - 0.12, hi + 0.15)
    fig.tight_layout()
    path = OUT / "11_params_vs_mpjpe.png"
    fig.savefig(path, dpi=160, facecolor="white", edgecolor="none")
    plt.close(fig)
    return path


def main():
    style()
    by_exp = load_by_exp()
    p9 = plot_09_mpjpe(by_exp)
    p10 = plot_10_depth_thermal(by_exp)
    p11 = plot_11_params_vs_mpjpe(by_exp)
    print(f"saved {p9}")
    print(f"saved {p10}")
    print(f"saved {p11}")


if __name__ == "__main__":
    main()

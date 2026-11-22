#!/usr/bin/env python3
"""Overlay val curves: full hybrids vs no-ForwardThermal ablation.

Reads log/hybrid{A,B,C,D}_{full,nofwd_full}.log
Writes figures/12_nofwd_vs_full_curves.png
"""
from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

LABELS = {"A": "Early", "B": "Trunk", "C": "Late", "D": "Map"}
LINE_RE = re.compile(
    r"epoch=(\d+)\s+train_loss=([0-9.]+)\s+val_mpjpe_cm=([0-9.]+)\s+"
    r"hand_depth_cm=([0-9.nan]+)\s+depth_cm=([0-9.nan]+)\s+recon=([0-9.nan]+)"
)


def parse_log(path: Path) -> dict[str, np.ndarray]:
    epochs, mpjpe, hand, depth = [], [], [], []
    if not path.exists():
        return {k: np.array([]) for k in ("epoch", "mpjpe", "hand", "depth")}
    for line in path.read_text(errors="ignore").splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        epochs.append(int(m.group(1)))
        mpjpe.append(float(m.group(3)))
        hand.append(float("nan") if m.group(4) == "nan" else float(m.group(4)))
        depth.append(float("nan") if m.group(5) == "nan" else float(m.group(5)))
    # keep last value per epoch if resumed/duplicated
    by_ep: dict[int, tuple] = {}
    for e, j, h, d in zip(epochs, mpjpe, hand, depth):
        by_ep[e] = (j, h, d)
    es = sorted(by_ep)
    return {
        "epoch": np.array(es, dtype=float),
        "mpjpe": np.array([by_ep[e][0] for e in es]),
        "hand": np.array([by_ep[e][1] for e in es]),
        "depth": np.array([by_ep[e][2] for e in es]),
    }


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


def main():
    style()
    letters = list("ABCD")
    full = {L: parse_log(ROOT / "log" / f"hybrid{L}_full.log") for L in letters}
    nofwd = {L: parse_log(ROOT / "log" / f"hybrid{L}_nofwd_full.log") for L in letters}
    xmax_nofwd = max((int(nofwd[L]["epoch"].max()) if len(nofwd[L]["epoch"]) else 0) for L in letters)
    # early window: cover nofwd + a bit of context; full still shown for whole run in right col via twin view
    early_xmax = max(40, xmax_nofwd + 5)

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.2), sharex="row", sharey="row")
    c_full = "#4c78a8"
    c_nofwd = "#e45756"

    for i, L in enumerate(letters):
        ax_m = axes[0, i]
        ax_h = axes[1, i]
        f, n = full[L], nofwd[L]

        # MPJPE — early window so ablation is readable
        if len(f["epoch"]):
            mask = f["epoch"] <= early_xmax
            ax_m.plot(f["epoch"][mask], f["mpjpe"][mask], color=c_full, lw=1.8, label="Full (+F)")
            # faint rest of full curve for context
            if (~mask).any():
                ax_m.plot(f["epoch"][~mask], f["mpjpe"][~mask], color=c_full, lw=1.0, alpha=0.25)
        if len(n["epoch"]):
            ax_m.plot(n["epoch"], n["mpjpe"], color=c_nofwd, lw=2.0, label="No Forward")
            bi = int(np.nanargmin(n["mpjpe"]))
            ax_m.scatter([n["epoch"][bi]], [n["mpjpe"][bi]], color=c_nofwd, s=28, zorder=3)

        ax_m.set_title(LABELS[L], fontweight="bold")
        ax_m.set_xlim(1, early_xmax)
        ax_m.grid(True, alpha=0.25)
        if i == 0:
            ax_m.set_ylabel("Val MPJPE (cm)")
            ax_m.legend(frameon=False, loc="upper right", fontsize=9)

        # Hand-depth MAE
        if len(f["epoch"]):
            mask = f["epoch"] <= early_xmax
            ax_h.plot(f["epoch"][mask], f["hand"][mask], color=c_full, lw=1.8, label="Full (+F)")
            if (~mask).any():
                ax_h.plot(f["epoch"][~mask], f["hand"][~mask], color=c_full, lw=1.0, alpha=0.25)
        if len(n["epoch"]):
            ax_h.plot(n["epoch"], n["hand"], color=c_nofwd, lw=2.0, label="No Forward")

        ax_h.set_xlim(1, early_xmax)
        ax_h.set_xlabel("Epoch")
        ax_h.grid(True, alpha=0.25)
        if i == 0:
            ax_h.set_ylabel("Hand-depth MAE (cm)")

    # annotate xmax for nofwd
    fig.suptitle(
        f"Full vs no-ForwardThermal ablation (nofwd through ep{xmax_nofwd}; "
        f"x-axis capped at {early_xmax})",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    out = OUT / "12_nofwd_vs_full_curves.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # Also a full-horizon MPJPE-only panel (full curves complete, nofwd overlaid)
    fig2, axes2 = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
    for i, L in enumerate(letters):
        ax = axes2[i]
        f, n = full[L], nofwd[L]
        if len(f["epoch"]):
            ax.plot(f["epoch"], f["mpjpe"], color=c_full, lw=1.6, label="Full (+F)")
            bi = int(np.nanargmin(f["mpjpe"]))
            ax.axhline(f["mpjpe"][bi], color=c_full, ls=":", lw=1.0, alpha=0.6)
        if len(n["epoch"]):
            ax.plot(n["epoch"], n["mpjpe"], color=c_nofwd, lw=2.0, label="No Forward")
        ax.set_title(LABELS[L], fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.set_ylabel("Val MPJPE (cm)")
            ax.legend(frameon=False, fontsize=9)
    fig2.suptitle("Full training horizon (nofwd still early)", fontsize=12, y=1.02)
    fig2.tight_layout()
    out2 = OUT / "12b_nofwd_vs_full_mpjpe_fullhorizon.png"
    fig2.savefig(out2, dpi=160, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()

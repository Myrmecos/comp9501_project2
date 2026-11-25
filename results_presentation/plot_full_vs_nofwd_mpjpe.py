#!/usr/bin/env python3
"""Official-test MPJPE (100% labeled): Full vs Nofwd for Early/Trunk/Late/Map.

Style matches the Depth Error panel in plot_09_10.py (light/dark blue grouped bars).
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

# Same palette as plot_09_10 Depth Error panel
C_FULL = "#88CCEE"
C_NOFWD = "#4c78a8"

HYBRIDS = [
    ("hybridA_full", "hybridA_nofwd_full", "Early"),
    ("hybridB_full", "hybridB_nofwd_full", "Trunk"),
    ("hybridC_full", "hybridC_nofwd_full", "Late"),
    ("hybridD_full", "hybridD_nofwd_full", "Map"),
]


def _fnum(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def main() -> Path:
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
    by_exp = {
        r["exp"]: r
        for r in json.loads((ROOT / "log" / "official_test_results.json").read_text())
        if r.get("status") == "ok"
    }

    labs = [lab for _, _, lab in HYBRIDS]
    full = [_fnum(by_exp[f]["test_mpjpe_cm"]) for f, _, _ in HYBRIDS]
    nofwd = [_fnum(by_exp[n]["test_mpjpe_cm"]) for _, n, _ in HYBRIDS]

    # Single panel sized like the left half of fig 10
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    xx = np.arange(len(labs))
    w = 0.36
    b1 = ax.bar(xx - w / 2, full, width=w, label="Full", color=C_FULL, zorder=3)
    b2 = ax.bar(xx + w / 2, nofwd, width=w, label="Nofwd", color=C_NOFWD, zorder=3)
    ax.set_xticks(xx)
    ax.set_xticklabels(labs)
    ax.set_ylabel("MPJPE (cm)")
    ax.set_title("Test MPJPE (100% labeled)")
    ax.set_facecolor("white")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, max(full + nofwd) * 1.2)
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.04,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#222",
            )

    fig.patch.set_facecolor("white")
    fig.tight_layout()
    path = OUT / "14_test_mpjpe_full_vs_nofwd.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"saved {path}")
    return path


if __name__ == "__main__":
    main()

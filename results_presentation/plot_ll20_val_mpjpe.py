#!/usr/bin/env python3
"""Val MPJPE at 20% labeled: Full vs Nofwd for Early/Trunk/Late/Map.

Values from the limited-label ablation table (Full / Nofwd).
Style matches 14_test_mpjpe_full_vs_nofwd.png (no gridlines).
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C_FULL = "#88CCEE"
C_NOFWD = "#4c78a8"

# 20% row from limited-label table (Full / Nofwd)
ROWS = [
    ("Early", 3.556, 3.880),
    ("Trunk", 3.778, 3.713),
    ("Late", 3.730, 3.886),
    ("Map", 3.799, 3.731),
]


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

    labs = [r[0] for r in ROWS]
    full = [r[1] for r in ROWS]
    nofwd = [r[2] for r in ROWS]

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    xx = np.arange(len(labs))
    w = 0.36
    b1 = ax.bar(xx - w / 2, full, width=w, label="Full", color=C_FULL, zorder=3)
    b2 = ax.bar(xx + w / 2, nofwd, width=w, label="Nofwd", color=C_NOFWD, zorder=3)
    ax.set_xticks(xx)
    ax.set_xticklabels(labs)
    ax.set_ylabel("MPJPE (cm)")
    ax.set_title("Test MPJPE (20% labeled)", pad=8)
    ax.set_facecolor("white")
    ax.set_ylim(0, max(full + nofwd) * 1.18)
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

    # Legend above the title so it never covers Early's value labels
    fig.legend(
        handles=[b1, b2],
        labels=["Full", "Nofwd"],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.12, 1.02),
        ncol=2,
        borderaxespad=0.0,
    )
    fig.patch.set_facecolor("white")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    path = OUT / "15_val_mpjpe_ll20_full_vs_nofwd.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"saved {path}")
    return path


if __name__ == "__main__":
    main()

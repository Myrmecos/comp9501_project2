#!/usr/bin/env python3
"""Collect physics_correct_ws micro Forward-vs-nofwd gaps."""
from __future__ import annotations

import json
import math
from pathlib import Path

SW = Path(__file__).resolve().parents[1]
OUT = SW / "log" / "MICRO_SCREEN.md"

CANDS = [
    ("P0", "P0", "P0_nofwd"),
    ("P1", "P1", "P1_nofwd"),
    ("P2", "P2", "P2_nofwd"),
    ("P3", "P3", "P3_nofwd"),
    ("P4", "P4", "P4_nofwd"),
]


def best(exp: str) -> dict | None:
    p = SW / "weights" / exp / "summary.json"
    if not p.exists():
        return None
    b = json.loads(p.read_text()).get("best") or {}
    return {
        "mpjpe": float(b.get("mpjpe_cm", float("nan"))),
        "hand": float(b.get("hand_depth_mae_cm", float("nan"))),
        "recon": float(b.get("recon_mae", float("nan"))),
        "epoch": b.get("epoch"),
    }


def main() -> None:
    lines = [
        "# physics_correct_ws micro screen",
        "",
        "gap = nofwd_mpjpe − full_mpjpe (positive ⇒ Forward helps).",
        "P0 residual control; P1/P2/P4 pure physics; P3 residual+phys/δ regularizers.",
        "",
        "| ID | full MPJPE | nofwd MPJPE | gap | full hand | full recon | status |",
        "|----|------------|-------------|-----|-----------|------------|--------|",
    ]
    best_gap, best_id = -1e9, None
    for cid, full_e, nofwd_e in CANDS:
        f, n = best(full_e), best(nofwd_e)
        gap = None
        if f and n and math.isfinite(f["mpjpe"]) and math.isfinite(n["mpjpe"]):
            gap = n["mpjpe"] - f["mpjpe"]
        if f is None and n is None:
            lines.append(f"| {cid} | — | — | — | — | — | pending |")
        else:
            fs = f"{f['mpjpe']:.3f}" if f else "—"
            ns = f"{n['mpjpe']:.3f}" if n else "—"
            gs = f"{gap:+.3f}" if gap is not None else "—"
            fh = f"{f['hand']:.3f}" if f and math.isfinite(f["hand"]) else "—"
            fr = f"{f['recon']:.3f}" if f and math.isfinite(f["recon"]) else "—"
            st = "done" if gap is not None else "partial"
            lines.append(f"| {cid} | {fs} | {ns} | {gs} | {fh} | {fr} | {st} |")
            if gap is not None and gap > best_gap:
                best_gap, best_id = gap, cid
    lines.append("")
    if best_id is not None:
        lines.append(f"**Best gap:** {best_id} = {best_gap:+.3f} cm")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()

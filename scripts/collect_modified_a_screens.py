#!/usr/bin/env python3
"""Summarize modified_A screen pairs: gap = nofwd_mpjpe - full_mpjpe (>0 => Forward helps)."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "log" / "MODIFIED_A_SCREEN.md"


def best(exp: str) -> dict | None:
    p = ROOT / "weights" / exp / "summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    b = d.get("best") or {}
    if not b:
        return None
    return {
        "mpjpe": float(b.get("mpjpe_cm", float("nan"))),
        "hand": float(b.get("hand_depth_mae_cm", float("nan"))),
        "recon": b.get("recon_mae"),
        "epoch": b.get("epoch"),
        "status": "done",
    }


def main() -> None:
    rows = []
    for sid in range(7):
        full = best(f"modifiedA_scr{sid}_full")
        nofwd = best(f"modifiedA_scr{sid}_nofwd")
        gap = None
        if full and nofwd and math.isfinite(full["mpjpe"]) and math.isfinite(nofwd["mpjpe"]):
            gap = nofwd["mpjpe"] - full["mpjpe"]
        rows.append({"sid": sid, "full": full, "nofwd": nofwd, "gap": gap})

    lines = [
        "# modified_A screen results",
        "",
        "gap = nofwd_mpjpe − full_mpjpe (positive ⇒ Forward helps)",
        "",
        "| screen | full MPJPE | nofwd MPJPE | gap | full hand | nofwd hand | status |",
        "|--------|------------|-------------|-----|-----------|------------|--------|",
    ]
    best_gap = -1e9
    best_sid = None
    for r in rows:
        f, n = r["full"], r["nofwd"]
        if f is None and n is None:
            lines.append(f"| S{r['sid']} | — | — | — | — | — | pending |")
            continue
        fs = f"{f['mpjpe']:.3f}" if f else "—"
        ns = f"{n['mpjpe']:.3f}" if n else "—"
        gs = f"{r['gap']:+.3f}" if r["gap"] is not None else "—"
        fh = f"{f['hand']:.3f}" if f and math.isfinite(f["hand"]) else "—"
        nh = f"{n['hand']:.3f}" if n and math.isfinite(n["hand"]) else "—"
        st = "done" if f and n else "partial"
        lines.append(f"| S{r['sid']} | {fs} | {ns} | {gs} | {fh} | {nh} | {st} |")
        if r["gap"] is not None and r["gap"] > best_gap:
            best_gap = r["gap"]
            best_sid = r["sid"]

    lines.append("")
    if best_sid is not None:
        lines.append(f"**Best screen:** S{best_sid} with gap={best_gap:+.3f} cm")
        if best_gap >= 0.08:
            lines.append("Meets target gap ≥ 0.08 cm → promote to full-budget train.")
        elif best_gap >= 0.05:
            lines.append("Gap ≥ 0.05 cm (soft pass) → can promote; prefer more screens if time.")
        else:
            lines.append("No screen yet meets ≥ 0.08 cm; continue grid / escalate arch.")
    lines.append("")
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT.read_text())
    (ROOT / "log" / "modified_a_screens.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

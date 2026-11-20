#!/usr/bin/env python3
"""Collect micro Forward-vs-nofwd gaps under small_ws/."""
from __future__ import annotations

import json
import math
from pathlib import Path

SW = Path(__file__).resolve().parents[1]
OUT = SW / "log" / "MICRO_SCREEN.md"

CANDS = [
    ("C0", "C0", "C0_nofwd"),
    ("C1", "C1", "C1_nofwd"),
    ("C2", "C2", "C2_nofwd"),
    ("C3", "C3", "C3_nofwd"),
    ("C4", "C4_ft", "C4_ft_nofwd"),
    ("C5", "C5", "C5_nofwd"),
    ("C6", "C6", "C6_nofwd"),
    ("C7", "C7", "C7_nofwd"),
    ("C8", "C8", "C8_nofwd"),
    ("C9", "C9", "C9_nofwd"),
    ("C10", "C10", "C10_nofwd"),
    ("C11", "C11", "C11_nofwd"),
    ("C13", "C13", "C13_nofwd"),
    ("C14", "C14", "C14_nofwd"),
    ("C15", "C15", "C15_nofwd"),
    ("C16", "C16", "C16_nofwd"),
]


def best(exp: str) -> dict | None:
    p = SW / "weights" / exp / "summary.json"
    if not p.exists():
        return None
    b = (json.loads(p.read_text()).get("best") or {})
    return {
        "mpjpe": float(b.get("mpjpe_cm", float("nan"))),
        "hand": float(b.get("hand_depth_mae_cm", float("nan"))),
        "epoch": b.get("epoch"),
    }


def main() -> None:
    lines = [
        "# small_ws micro screen",
        "",
        "gap = nofwd_mpjpe − full_mpjpe (positive ⇒ Forward helps). Split: 2k/500 frames.",
        "Wave1 = C0–C5 (30 ep); Wave2 = C6–C11 (20 ep). Original HybridA weights untouched.",
        "",
        "| ID | full MPJPE | nofwd MPJPE | gap | full hand | nofwd hand | status |",
        "|----|------------|-------------|-----|-----------|------------|--------|",
    ]
    best_gap, best_id = -1e9, None
    rows = []
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
            nh = f"{n['hand']:.3f}" if n and math.isfinite(n["hand"]) else "—"
            st = "done" if gap is not None else "partial"
            lines.append(f"| {cid} | {fs} | {ns} | {gs} | {fh} | {nh} | {st} |")
            if gap is not None and gap > best_gap:
                best_gap, best_id = gap, cid
        rows.append({"id": cid, "full": f, "nofwd": n, "gap": gap})
    lines.append("")
    if best_id is not None:
        lines.append(f"**Best:** {best_id} gap={best_gap:+.3f} cm")
        if best_gap >= 0.05:
            lines.append("→ promote to larger split (`limited_label_f20` / full).")
        else:
            lines.append("No candidate yet ≥ 0.05 cm; iterate structures/λ.")
    lines.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    (SW / "log" / "micro_screens.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()

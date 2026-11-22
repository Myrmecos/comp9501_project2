#!/usr/bin/env python3
"""Collect modified_A(P4) + modified_B/C/D Forward vs nofwd across labeled fractions."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PC = ROOT.parent / "physics_correct_ws"
OUT = ROOT / "log" / "MODIFIED_BCD_FRAC.md"

TRAIN_N = {"05": 1383, "10": 2766, "20": 5531, "60": 16592}


def best(art: Path, exp: str) -> float | None:
    p = art / "weights" / exp / "summary.json"
    if not p.exists():
        return None
    v = float((json.loads(p.read_text()).get("best") or {}).get("mpjpe_cm", float("nan")))
    return v if math.isfinite(v) else None


def main() -> None:
    lines = [
        "# Modified A/B/C/D Forward vs nofwd by labeled fraction",
        "",
        "Recipe: residual OFF, λ_recon 0.02→0, keep_depth_attached, mask/inv-skip.",
        "gap = nofwd − Forward (positive ⇒ Forward helps).",
        "",
    ]
    # A from physics_correct_ws P4_fXX
    lines += [
        "## modified_A (P4)",
        "",
        "| labeled% | train N | Forward | nofwd | gap | status |",
        "|----------|---------|---------|-------|-----|--------|",
    ]
    for key, pct in [("05", 5), ("10", 10), ("20", 20), ("60", 60)]:
        f, n = best(PC, f"P4_f{key}"), best(PC, f"P4_f{key}_nofwd")
        if f is None and n is None:
            lines.append(f"| {pct}% | {TRAIN_N[key]} | — | — | — | pending |")
        else:
            gap = (n - f) if f is not None and n is not None else None
            fs = f"{f:.3f}" if f is not None else "—"
            ns = f"{n:.3f}" if n is not None else "—"
            gs = f"{gap:+.3f}" if gap is not None else "—"
            st = "done" if gap is not None else "partial"
            lines.append(f"| {pct}% | {TRAIN_N[key]} | {fs} | {ns} | {gs} | {st} |")

    for letter in ("B", "C", "D", "C2"):
        lines += [
            "",
            f"## modified_{letter}" + (" (residual ON)" if letter=="C2" else ""),
            "",
            "| labeled% | train N | Forward | nofwd | gap | status |",
            "|----------|---------|---------|-------|-----|--------|",
        ]
        for key, pct in [("05", 5), ("10", 10), ("20", 20), ("60", 60)]:
            f, n = best(ROOT, f"M{letter}_f{key}"), best(ROOT, f"M{letter}_f{key}_nofwd")
            if f is None and n is None:
                lines.append(f"| {pct}% | {TRAIN_N[key]} | — | — | — | pending |")
            else:
                gap = (n - f) if f is not None and n is not None else None
                fs = f"{f:.3f}" if f is not None else "—"
                ns = f"{n:.3f}" if n is not None else "—"
                gs = f"{gap:+.3f}" if gap is not None else "—"
                st = "done" if gap is not None else "partial"
                lines.append(f"| {pct}% | {TRAIN_N[key]} | {fs} | {ns} | {gs} | {st} |")

    # micro appendix
    lines += ["", "## Micro screen (prerequisite)", ""]
    micro = ROOT / "log" / "MODIFIED_BCD_MICRO.md"
    if micro.exists():
        lines.append(micro.read_text())
    else:
        lines.append("(pending)")

    OUT.write_text("\n".join(lines) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()

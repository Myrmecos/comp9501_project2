#!/usr/bin/env python3
"""Collect P4 Forward vs nofwd across labeled fractions."""
from __future__ import annotations

import json
import math
from pathlib import Path

SW = Path(__file__).resolve().parents[1]
OUT = SW / "log" / "LABEL_FRAC_SCREEN.md"
OUT_JSON = SW / "log" / "label_frac_results.json"

FRACS = [("05", 5), ("10", 10), ("20", 20), ("60", 60)]


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
    rows = []
    lines = [
        "# P4 Forward vs nofwd by labeled fraction",
        "",
        "Recipe: `modified_A`, residual OFF, λ_recon 0.02→0, mask/inv-skip.",
        "Splits: `limited_label_fXX_coupled.json`. gap = nofwd − Forward (positive ⇒ Forward helps).",
        "",
        "| labeled% | train N | Forward MPJPE | nofwd MPJPE | gap | Forward hand | nofwd hand | status |",
        "|----------|---------|---------------|-------------|-----|--------------|------------|--------|",
    ]
    train_n = {"05": 1383, "10": 2766, "20": 5531, "60": 16592}
    for key, pct in FRACS:
        f = best(f"P4_f{key}")
        n = best(f"P4_f{key}_nofwd")
        gap = None
        if f and n and math.isfinite(f["mpjpe"]) and math.isfinite(n["mpjpe"]):
            gap = n["mpjpe"] - f["mpjpe"]
        if f is None and n is None:
            lines.append(f"| {pct}% | {train_n[key]} | — | — | — | — | — | pending |")
        else:
            fs = f"{f['mpjpe']:.3f}" if f else "—"
            ns = f"{n['mpjpe']:.3f}" if n else "—"
            gs = f"{gap:+.3f}" if gap is not None else "—"
            fh = f"{f['hand']:.3f}" if f and math.isfinite(f["hand"]) else "—"
            nh = f"{n['hand']:.3f}" if n and math.isfinite(n["hand"]) else "—"
            st = "done" if gap is not None else "partial"
            lines.append(f"| {pct}% | {train_n[key]} | {fs} | {ns} | {gs} | {fh} | {nh} | {st} |")
        rows.append({"frac": pct, "train_n": train_n[key], "forward": f, "nofwd": n, "gap": gap})

    # Also note older HybridA SSL→FT numbers for reference
    lines += [
        "",
        "## Reference: older HybridA ll (SSL→FT vs nofwd, main `weights/`)",
        "",
        "| labeled% | FT (Forward path) | nofwd | gap |",
        "|----------|-------------------|-------|-----|",
    ]
    wroot = SW.parent / "weights"
    for pct in (5, 10, 20, 60):
        def mpj(exp):
            p = wroot / exp / "summary.json"
            if not p.exists():
                return None
            return float((json.loads(p.read_text()).get("best") or {}).get("mpjpe_cm", float("nan")))

        ft = mpj(f"hybridA_ll{pct:02d}_ft")
        nf = mpj(f"hybridA_ll{pct:02d}_nofwd")
        if ft is not None and nf is not None and math.isfinite(ft) and math.isfinite(nf):
            lines.append(f"| {pct}% | {ft:.3f} | {nf:.3f} | {nf-ft:+.3f} |")
        else:
            lines.append(f"| {pct}% | {ft} | {nf} | — |")

    lines += [
        "",
        "Note: HybridA reference uses a different model/protocol (SSL pretrain → FT); not directly comparable to P4.",
        "",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    OUT_JSON.write_text(json.dumps(rows, indent=2) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()

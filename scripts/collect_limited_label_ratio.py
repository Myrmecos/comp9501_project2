#!/usr/bin/env python3
"""Collect Early limited-label Full(ft) vs Nofwd vs ResNet/hybridA full references.

Writes:
  log/LIMITED_LABEL_RATIO_RESULTS.md
  log/limited_label_ratio_results.json
  results_presentation/figures/13_limited_label_ratio_mpjpe.png (if any ft/nofwd bests exist)
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
OUT_MD = ROOT / "log" / "LIMITED_LABEL_RATIO_RESULTS.md"
OUT_JSON = ROOT / "log" / "limited_label_ratio_results.json"
OUT_FIG = ROOT / "results_presentation" / "figures" / "13_limited_label_ratio_mpjpe.png"

FRACS = [5, 10, 20, 40, 60]
LINE_RE = re.compile(
    r"epoch=(\d+)\s+.*val_mpjpe_cm=([0-9.]+).*recon=([0-9.nan]+)"
)


def _best_from_summary(exp: str) -> dict | None:
    p = WEIGHTS / exp / "summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    best = d.get("best") or {}
    if not best:
        return None
    return {
        "exp": exp,
        "status": "done",
        "best_epoch": best.get("epoch"),
        "val_mpjpe_cm": best.get("mpjpe_cm"),
        "val_hand_depth_mae_cm": best.get("hand_depth_mae_cm"),
        "val_recon_mae": best.get("recon_mae"),
        "early_stop_metric": d.get("early_stop_metric"),
    }


def _progress_from_log(exp: str) -> dict:
    log = ROOT / "log" / f"{exp}.log"
    pidf = ROOT / "log" / f"{exp}.pid"
    running = False
    if pidf.exists():
        try:
            import os as _os

            running = _os.path.exists(f"/proc/{pidf.read_text().strip()}")
        except Exception:
            running = False
    if not log.exists():
        return {"exp": exp, "status": "missing"}
    last = None
    for line in log.read_text(errors="ignore").splitlines():
        m = LINE_RE.search(line)
        if m:
            last = {
                "epoch": int(m.group(1)),
                "val_mpjpe_cm": float(m.group(2)),
                "recon": None if m.group(3) == "nan" else float(m.group(3)),
            }
        if "done best_" in line or line.endswith("done") or "done best_mpjpe" in line:
            running = False
    if last is None:
        return {"exp": exp, "status": "running" if running else "no_epochs"}
    return {
        "exp": exp,
        "status": "running" if running else "done_no_summary",
        "last_epoch": last["epoch"],
        "last_val_mpjpe_cm": last["val_mpjpe_cm"],
        "last_recon": last["recon"],
    }


def collect_exp(exp: str) -> dict:
    best = _best_from_summary(exp)
    if best:
        return best
    return _progress_from_log(exp)


def main() -> None:
    rows = []
    for pct in FRACS:
        for kind in ("ft", "nofwd", "ssl"):
            rows.append({"coupled_pct": pct, "kind": kind, **collect_exp(f"hybridA_ll{pct:02d}_{kind}")})
    for exp, label in (("resnet_full", "resnet_full"), ("hybridA_full", "hybridA_full")):
        r = collect_exp(exp)
        r["coupled_pct"] = 100
        r["kind"] = label
        rows.append(r)

    OUT_JSON.write_text(json.dumps(rows, indent=2) + "\n")

    lines = [
        "# Limited-label ratio results (Early / HybridA)",
        "",
        "| coupled% | kind | status | val MPJPE | epoch | notes |",
        "|----------|------|--------|-----------|-------|-------|",
    ]
    for r in rows:
        mpj = r.get("val_mpjpe_cm", r.get("last_val_mpjpe_cm"))
        ep = r.get("best_epoch", r.get("last_epoch"))
        mpjs = f"{mpj:.3f}" if isinstance(mpj, (int, float)) and math.isfinite(float(mpj)) else "—"
        lines.append(
            f"| {r.get('coupled_pct')} | {r.get('kind')} | {r.get('status')} | {mpjs} | {ep if ep is not None else '—'} | {r.get('exp')} |"
        )
    lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_JSON}")

    # Plot when we have at least one finished ft or nofwd point
    xs_ft, ys_ft, xs_nf, ys_nf = [], [], [], []
    for r in rows:
        if r.get("kind") == "ft" and isinstance(r.get("val_mpjpe_cm"), (int, float)):
            xs_ft.append(r["coupled_pct"])
            ys_ft.append(r["val_mpjpe_cm"])
        if r.get("kind") == "nofwd" and isinstance(r.get("val_mpjpe_cm"), (int, float)):
            xs_nf.append(r["coupled_pct"])
            ys_nf.append(r["val_mpjpe_cm"])

    ref_a = next((r for r in rows if r.get("kind") == "hybridA_full" and r.get("val_mpjpe_cm") is not None), None)
    ref_r = next((r for r in rows if r.get("kind") == "resnet_full" and r.get("val_mpjpe_cm") is not None), None)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if xs_ft:
        ax.plot(xs_ft, ys_ft, "o-", color="#4c78a8", lw=2, label="Full (SSL→FT)")
    if xs_nf:
        ax.plot(xs_nf, ys_nf, "s--", color="#e45756", lw=2, label="Nofwd")
    if ref_a and isinstance(ref_a.get("val_mpjpe_cm"), (int, float)):
        ax.axhline(ref_a["val_mpjpe_cm"], color="#4c78a8", ls=":", alpha=0.7, label="hybridA_full (100%)")
    if ref_r and isinstance(ref_r.get("val_mpjpe_cm"), (int, float)):
        ax.axhline(ref_r["val_mpjpe_cm"], color="#59a14f", ls=":", alpha=0.8, label="resnet_full (100%)")
    ax.set_xlabel("Coupled label fraction (%)")
    ax.set_ylabel("Val MPJPE (cm)")
    ax.set_title("Early: Full vs Nofwd vs full-data references")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=160)
    plt.close(fig)
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()

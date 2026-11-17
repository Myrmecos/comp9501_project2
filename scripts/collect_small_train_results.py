#!/usr/bin/env python3
"""Collect small-split parallel training summaries into EXPERIMENT_SUMMARY.md."""
from __future__ import annotations

import json
from pathlib import Path

EXPS = [
    ("vanilla (full TAPOR)", "vanilla_maskfix_small"),
    ("A", "hybridA_maskfix_small"),
    ("B", "hybridB_maskfix_small"),
    ("C", "hybridC_maskfix_small"),
    ("D", "hybridD_maskfix_small"),
]

PREV = {
    "vanilla_full_small": "vanilla (prev)",
    "hybridA_full_small": "A (prev)",
    "hybridB_full_small": "B (prev)",
    "hybridC_full_small": "C (prev)",
    "hybridD_full_small": "D (prev)",
}


def _row(model: str, exp: str, path: Path) -> str:
    data = json.loads(path.read_text())
    best = data.get("best") or {}
    return (
        "| {model} | {exp} | {epoch} | {mpjpe:.3f} | {hand:.3f} | {depth:.3f} | {recon:.3f} | "
        "{params:.3f} | {elapsed:.1f} | `weights/{exp}/best.pt` |"
    ).format(
        model=model,
        exp=exp,
        epoch=best.get("epoch", "-"),
        mpjpe=float(best.get("mpjpe_cm", float("nan"))),
        hand=float(best.get("hand_depth_mae_cm", float("nan"))),
        depth=float(best.get("depth_mae_cm", float("nan"))),
        recon=float(best.get("recon_mae", float("nan"))),
        params=float(data.get("params_m", float("nan"))),
        elapsed=float(data.get("elapsed_sec", float("nan"))),
    )


def main():
    root = Path(".")
    lines = [
        "# Small-split training (adaptive mask + hand-depth focus + loss rebalance)",
        "",
        "Changes vs prior full-TAPOR run:",
        "- Adaptive warm-hand mask (percentile + ambient delta); recon/depth losses on hand only",
        "- Hand-focused depth GT + strong `hand_depth` loss; weak masked map depth",
        "- Two-phase loss schedule; stop-grad physics→pose in phase 2",
        "- Full TAPOR baseline uses `num_history=4` + ambient-normalized inputs",
        "",
        "Five models trained in parallel on one A100 (foreign PIDs untouched).",
        "",
        "| Model | Exp ID | Best epoch | Val MPJPE (cm) | Hand-depth MAE (cm) | Depth MAE (cm) | Recon MAE | Params (M) | Elapsed (s) | Best ckpt |",
        "|-------|--------|------------|----------------|---------------------|----------------|-----------|------------|-------------|-----------|",
    ]
    for model, exp in EXPS:
        summary_path = root / "log" / f"{exp}_summary.json"
        alt = root / "weights" / exp / "summary.json"
        path = summary_path if summary_path.exists() else alt
        if not path.exists():
            lines.append(f"| {model} | {exp} | - | - | - | - | - | - | - | missing |")
            continue
        lines.append(_row(model, exp, path))

    lines.append("")
    lines.append("## Previous run (for comparison)")
    lines.append("")
    lines.append("| Model | Exp ID | Val MPJPE (cm) | Depth MAE (cm) | Recon MAE |")
    lines.append("|-------|--------|----------------|----------------|-----------|")
    for exp, model in PREV.items():
        path = root / "log" / f"{exp}_summary.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        best = data.get("best") or {}
        lines.append(
            "| {model} | {exp} | {mpjpe:.3f} | {depth:.3f} | {recon:.3f} |".format(
                model=model,
                exp=exp,
                mpjpe=float(best.get("mpjpe_cm", float("nan"))),
                depth=float(best.get("depth_mae_cm", float("nan"))),
                recon=float(best.get("recon_mae", float("nan"))),
            )
        )

    lines.append("")
    lines.append("TensorBoard logs: `log/tb/{exp_id}/`")
    lines.append("")
    split = root / "data/tapor/small_split.json"
    if split.exists():
        payload = json.loads(split.read_text())
        lines.append(
            f"Split: {len(payload.get('train', []))} train pickles, "
            f"{len(payload.get('val', []))} val pickles (seed={payload.get('seed')})."
        )
    out = root / "log" / "EXPERIMENT_SUMMARY.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()

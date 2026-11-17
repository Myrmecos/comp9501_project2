#!/usr/bin/env python3
"""Append MediaPipe3d / NanoTapor small-split results into EXPERIMENT_SUMMARY.md."""
from __future__ import annotations

import json
from pathlib import Path

EXPS = [
    ("MediaPipe3d (paper)", "mediapipe_small"),
    ("NanoTapor (paper)", "nanotapor_small"),
    ("full TAPOR (ours, maskfix)", "vanilla_maskfix_small"),
    ("Hybrid C (best physics)", "hybridC_maskfix_small"),
]


def main():
    root = Path(".")
    lines = [
        "",
        "## TAPOR paper baselines on the same small split",
        "",
        "Trained from scratch on identical 12/4 pickle small split (seed=0).",
        "MediaPipe3d: BlazeHandLandmark, thermal upsampled 32×24→256×256, 1-ch.",
        "NanoTapor: official EdgeTaporStudent CNN (no KD teacher features on this run).",
        "",
        "| Model | Exp ID | Best epoch | Val MPJPE (cm) | Params (M) | Elapsed (s) | Best ckpt |",
        "|-------|--------|------------|----------------|------------|-------------|-----------|",
    ]
    for model, exp in EXPS:
        path = root / "log" / f"{exp}_summary.json"
        if not path.exists():
            path = root / "weights" / exp / "summary.json"
        if not path.exists():
            lines.append(f"| {model} | {exp} | - | - | - | - | missing |")
            continue
        data = json.loads(path.read_text())
        best = data.get("best") or {}
        lines.append(
            "| {model} | {exp} | {epoch} | {mpjpe:.3f} | {params:.3f} | {elapsed:.1f} | `weights/{exp}/best.pt` |".format(
                model=model,
                exp=exp,
                epoch=best.get("epoch", "-"),
                mpjpe=float(best.get("mpjpe_cm", float("nan"))),
                params=float(data.get("params_m", float("nan"))),
                elapsed=float(data.get("elapsed_sec", float("nan"))),
            )
        )
    out = root / "log" / "EXPERIMENT_SUMMARY.md"
    prev = out.read_text() if out.exists() else ""
    # Replace previous paper-baselines section if present
    marker = "\n## TAPOR paper baselines on the same small split\n"
    if marker in prev:
        prev = prev.split(marker)[0].rstrip() + "\n"
    out.write_text(prev + "\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()

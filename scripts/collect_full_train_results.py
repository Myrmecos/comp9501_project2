#!/usr/bin/env python3
"""Collect full-dataset training summaries into log/FULL_EXPERIMENT_SUMMARY.md."""
from __future__ import annotations

import json
from pathlib import Path

EXPS = [
    ("Hand3d (paper)", "hand3d_full"),
    ("MediaPipe3d (paper)", "mediapipe_full"),
    ("NanoTapor (paper)", "nanotapor_full"),
    ("full TAPOR (base1)", "vanilla_full"),
    ("Hybrid A (full-power)", "hybridA_full"),
    ("Hybrid B (full-power)", "hybridB_full"),
    ("Hybrid C (full-power)", "hybridC_full"),
    ("Hybrid D (full-power)", "hybridD_full"),
]


def main():
    root = Path(".")
    lines = [
        "# Full-dataset parallel training summary",
        "",
        "Official TAPOR Trainset file list + 80/10/10 frame split (`data/tapor/full_split.json`).",
        "Hybrids A–D: spatial_ch=128, kp_dim=96, inverse_base=64 + adaptive mask / hand-depth losses.",
        "TensorBoard: `log/tb/{exp_id}/` · best weights: `weights/{exp_id}/best.pt`",
        "",
        "| Model | Exp ID | Best epoch | Val MPJPE (cm) | Hand-depth MAE (cm) | Depth MAE (cm) | Recon MAE | Params (M) | Elapsed (s) | Best ckpt |",
        "|-------|--------|------------|----------------|---------------------|----------------|-----------|------------|-------------|-----------|",
    ]
    for model, exp in EXPS:
        path = root / "log" / f"{exp}_summary.json"
        if not path.exists():
            path = root / "weights" / exp / "summary.json"
        if not path.exists():
            lines.append(f"| {model} | {exp} | - | - | - | - | - | - | - | missing |")
            continue
        data = json.loads(path.read_text())
        best = data.get("best") or {}
        lines.append(
            "| {model} | {exp} | {epoch} | {mpjpe:.3f} | {hand:.3f} | {depth:.3f} | {recon:.3f} | "
            "{params:.3f} | {elapsed:.1f} | `weights/{exp}/best.pt` |".format(
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
        )
    out = root / "log" / "FULL_EXPERIMENT_SUMMARY.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()

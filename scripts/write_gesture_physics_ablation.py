#!/usr/bin/env python3
"""Build GESTURE_PHYSICS_ABLATION.md from gesture_*_summary.json + pose summaries."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [
    ("A", "hybridA_full", "hybridA_nofwd_full"),
    ("B", "hybridB_full", "hybridB_nofwd_full"),
    ("C", "hybridC_full", "hybridC_nofwd_full"),
    ("D", "hybridD_full", "hybridD_nofwd_full"),
]


def load_gesture(exp_id: str) -> dict | None:
    p = ROOT / "log" / f"{exp_id}_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def pose_mpjpe(pose_exp: str) -> float | None:
    # prefer official test
    ot = ROOT / "log" / "official_test_results.json"
    if ot.exists():
        for r in json.loads(ot.read_text()):
            if r.get("exp") == pose_exp or r.get("exp_id") == pose_exp:
                v = r.get("mpjpe_cm")
                if v is not None:
                    return float(v)
    sp = ROOT / "weights" / pose_exp / "summary.json"
    if sp.exists():
        b = (json.loads(sp.read_text()).get("best") or {})
        v = b.get("mpjpe_cm")
        if v is not None:
            return float(v)
    g = load_gesture(f"gesture_{pose_exp}")
    if g and g.get("pose_val_mpjpe_cm") is not None:
        return float(g["pose_val_mpjpe_cm"])
    return None


def fmt(x, nd=2):
    if x is None:
        return "—"
    return f"{float(x):.{nd}f}"


def main() -> None:
    lines = [
        "# Gesture × physics ablation",
        "",
        "Frozen pose → `GestureClassifier` on 21×3 joints (`plug_thermal_map=false`).",
        "Split: `data/tapor/gesture_split.json`. Same protocol as `GESTURE_RESULTS.md`.",
        "",
        "**Primary question:** does Forward/physics help gesture?",
        "`Δ Acc = Acc(full) − Acc(nofwd)` (positive ⇒ Forward helps).",
        "",
        "Pose MPJPE columns are **val** best from pose `summary.json` (cm).",
        "",
        "| Model | Acc full | Acc nofwd | Δ Acc | F1 full | F1 nofwd | Δ F1 | Pose MPJPE full | Pose MPJPE nofwd |",
        "|-------|----------|-----------|-------|---------|----------|------|-----------------|------------------|",
    ]
    deltas = []
    for model, full_e, nofwd_e in PAIRS:
        gf = load_gesture(f"gesture_{full_e}")
        gn = load_gesture(f"gesture_{nofwd_e}")
        af = gf.get("test_acc") if gf else None
        an = gn.get("test_acc") if gn else None
        ff = gf.get("test_macro_f1") if gf else None
        fn = gn.get("test_macro_f1") if gn else None
        dacc = (af - an) if af is not None and an is not None else None
        df1 = (ff - fn) if ff is not None and fn is not None else None
        if dacc is not None:
            deltas.append(dacc)
        lines.append(
            f"| {model} | {fmt(af)} | {fmt(an)} | {fmt(dacc)} | {fmt(ff)} | {fmt(fn)} | {fmt(df1)} "
            f"| {fmt(pose_mpjpe(full_e), 4)} | {fmt(pose_mpjpe(nofwd_e), 4)} |"
        )
    lines.append("")
    if deltas:
        mean_d = sum(deltas) / len(deltas)
        lines.append(f"**Mean Δ Acc (A–D):** {mean_d:+.2f} pp over {len(deltas)} pairs.")
        if mean_d > 0.5:
            lines.append("Forward/physics shows a **positive** gesture transfer on average.")
        elif mean_d < -0.5:
            lines.append("Forward/physics **hurts** gesture on average under joints-only protocol.")
        else:
            lines.append("Forward/physics is **≈ neutral** for gesture (within ~0.5 pp).")
    else:
        lines.append("Paired results incomplete — re-run after nofwd gesture heads finish.")

    # Fair baselines
    lines += [
        "",
        "## Fair baselines (retrained)",
        "",
        "| Backbone | Exp | Test Acc | Test F1 | Best ep | Note |",
        "|----------|-----|----------|---------|---------|------|",
    ]
    for eid, note in [
        ("gesture_vanilla_full_v2", "retrain; old ep2 run preserved"),
        ("gesture_gt_joints_v2", "retrain; old ep1 run preserved"),
        ("gesture_vanilla_full", "legacy undertrained"),
        ("gesture_gt_joints", "legacy undertrained"),
        ("gesture_nanotapor_full", "baseline"),
        ("gesture_mediapipe_full", "baseline"),
        ("gesture_hand3d_full", "baseline"),
    ]:
        g = load_gesture(eid)
        if not g:
            lines.append(f"| — | {eid} | — | — | — | missing |")
            continue
        lines.append(
            f"| {g.get('backbone')} | {eid} | {fmt(g.get('test_acc'))} | {fmt(g.get('test_macro_f1'))} "
            f"| {g.get('best_epoch')} | {note} |"
        )

    lines += [
        "",
        "Raw gesture summaries: `log/gesture_*_summary.json`.",
        "Full table: `log/GESTURE_RESULTS.md`.",
        "",
    ]
    out = ROOT / "log" / "GESTURE_PHYSICS_ABLATION.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()

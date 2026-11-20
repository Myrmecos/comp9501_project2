#!/usr/bin/env python3
"""Build scarce-label splits for TAP3D-style SSL pretrain → finetune.

From full_split_noleak.json train pool:
  - coupled fraction: thermal + depth + keypoints (Stage 2 Full + Nofwd)
  - remainder: thermal-only / unlabeled (Stage 1 Full SSL pretrain)

Emits per frac:
  limited_label_f{pct}_ssl.json      — train = unlabeled
  limited_label_f{pct}_coupled.json  — train = coupled
  limited_label_f{pct}_meta.json     — both lists + audit counts

Also writes log/LIMITED_LABEL_SPLIT_AUDIT.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _frame_key(entry: dict) -> tuple[str, int]:
    return (str(entry["rel"]), int(entry["frame"]))


def _with_flag(entries: list[dict], pose_labeled: bool) -> list[dict]:
    return [
        {"rel": e["rel"], "frame": int(e["frame"]), "pose_labeled": bool(pose_labeled)}
        for e in entries
    ]


def _pct_tag(frac: float) -> str:
    return f"{int(round(frac * 100)):02d}"


def build_for_frac(
    base: dict,
    base_path: Path,
    frac: float,
    seed: int,
) -> tuple[dict, dict, dict, dict]:
    """Return (ssl_split, coupled_split, meta, checks)."""
    train_pool = list(base["train"])
    n = len(train_pool)
    n_labeled = int(math.ceil(frac * n))
    n_labeled = max(0, min(n, n_labeled))

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    labeled_idx = set(int(i) for i in order[:n_labeled])

    coupled_raw = [train_pool[i] for i in range(n) if i in labeled_idx]
    unlabeled_raw = [train_pool[i] for i in range(n) if i not in labeled_idx]
    # Stable order by (rel, frame) for reproducibility of file contents beyond RNG
    coupled_raw.sort(key=_frame_key)
    unlabeled_raw.sort(key=_frame_key)

    coupled = _with_flag(coupled_raw, True)
    unlabeled = _with_flag(unlabeled_raw, False)
    val = _with_flag(base.get("val") or [], True)
    test = _with_flag(base.get("test") or [], True)

    common = {
        "seed": seed,
        "labeled_frac": float(frac),
        "base_split": str(base_path.as_posix()),
        "protocol": "tap3d_pretrain_finetune",
        "skip_start_frames": base.get("skip_start_frames"),
        "source": (
            f"Subsample of {base_path.name} train frames; "
            f"coupled_frac={frac}, seed={seed}"
        ),
        "train_files": base.get("train_files"),
        "val_files": base.get("val_files"),
        "dropped_basename_collisions": base.get("dropped_basename_collisions"),
        "audit": base.get("audit"),
        "val": val,
        "test": test,
    }

    ssl = {
        **common,
        "split_role": "ssl_pretrain",
        "train": unlabeled,
        "counts": {
            "train_frames": len(unlabeled),
            "train_unlabeled": len(unlabeled),
            "train_labeled": 0,
            "val_frames": len(val),
            "test_frames": len(test),
            "base_train_frames": n,
            "labeled_frac": float(frac),
        },
    }

    coupled_split = {
        **common,
        "split_role": "coupled_finetune",
        "train": coupled,
        "counts": {
            "train_frames": len(coupled),
            "train_labeled": len(coupled),
            "train_unlabeled": 0,
            "val_frames": len(val),
            "test_frames": len(test),
            "base_train_frames": n,
            "labeled_frac": float(frac),
        },
    }

    meta = {
        "seed": seed,
        "labeled_frac": float(frac),
        "base_split": str(base_path.as_posix()),
        "protocol": "tap3d_pretrain_finetune",
        "skip_start_frames": base.get("skip_start_frames"),
        "source": common["source"],
        "train_files": base.get("train_files"),
        "val_files": base.get("val_files"),
        "train_coupled": coupled,
        "train_unlabeled": unlabeled,
        "val": val,
        "test": test,
        "counts": {
            "train_labeled": len(coupled),
            "train_unlabeled": len(unlabeled),
            "train_total": n,
            "val_frames": len(val),
            "test_frames": len(test),
            "labeled_frac": float(frac),
        },
        "dropped_basename_collisions": base.get("dropped_basename_collisions"),
        "audit": base.get("audit"),
    }

    c_keys = {_frame_key(e) for e in coupled}
    u_keys = {_frame_key(e) for e in unlabeled}
    v_keys = {_frame_key(e) for e in val}
    base_keys = {_frame_key(e) for e in train_pool}
    train_files = {e["rel"] for e in coupled + unlabeled}
    val_files = {e["rel"] for e in val}

    checks = {
        "n_labeled": len(coupled),
        "n_unlabeled": len(unlabeled),
        "n_base": n,
        "expected_labeled": n_labeled,
        "intersection_empty": len(c_keys & u_keys) == 0,
        "union_covers_base": (c_keys | u_keys) == base_keys,
        "val_unchanged": len(val) == len(base.get("val") or []),
        "train_val_file_disjoint": len(train_files & val_files) == 0,
        "all_coupled_flag_true": all(e["pose_labeled"] for e in coupled),
        "all_unlabeled_flag_false": all(not e["pose_labeled"] for e in unlabeled),
        "all_val_flag_true": all(e["pose_labeled"] for e in val),
        "val_overlap_train_frames": len((c_keys | u_keys) & v_keys),
    }
    return ssl, coupled_split, meta, checks


def write_audit(path: Path, rows: list[dict], seed: int, base_path: Path) -> None:
    lines = [
        "# Limited-label split audit",
        "",
        "## Protocol (TAP3D-style)",
        "",
        "- **Stage 1 (Full SSL pretrain):** `limited_label_f*_ssl.json` — thermal-only; recon (+ mask).",
        "- **Stage 2 (Full finetune):** `limited_label_f*_coupled.json` — resume Stage 1; depth + keypoints.",
        "- **Nofwd:** same `*_coupled.json` from scratch; no ForwardThermal / no Stage 1.",
        "- Val stays fully coupled (inherited from noleak).",
        "",
        f"- Base split: `{base_path.as_posix()}`",
        f"- RNG seed: `{seed}`",
        "",
        "## Per-fraction checks",
        "",
        "| frac | coupled | unlabeled | base | ∩ empty | ∪=base | val n | train↔val files disjoint | ok |",
        "|------|---------|-----------|------|---------|--------|-------|--------------------------|----|",
    ]
    for r in rows:
        ok = all(
            [
                r["intersection_empty"],
                r["union_covers_base"],
                r["val_unchanged"],
                r["train_val_file_disjoint"],
                r["n_labeled"] == r["expected_labeled"],
                r["all_coupled_flag_true"],
                r["all_unlabeled_flag_false"],
                r["all_val_flag_true"],
                r["val_overlap_train_frames"] == 0,
            ]
        )
        lines.append(
            f"| {r['frac']:.2f} | {r['n_labeled']} | {r['n_unlabeled']} | {r['n_base']} | "
            f"{r['intersection_empty']} | {r['union_covers_base']} | {r['val_n']} | "
            f"{r['train_val_file_disjoint']} | {'PASS' if ok else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Files written",
            "",
        ]
    )
    for r in rows:
        pct = r["pct"]
        lines.append(f"- `data/tapor/limited_label_f{pct}_ssl.json`")
        lines.append(f"- `data/tapor/limited_label_f{pct}_coupled.json`")
        lines.append(f"- `data/tapor/limited_label_f{pct}_meta.json`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=Path("data/tapor/full_split_noleak.json"))
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("data/tapor"))
    ap.add_argument("--audit-out", type=Path, default=Path("log/LIMITED_LABEL_SPLIT_AUDIT.md"))
    args = ap.parse_args()

    base_path = args.base if args.base.is_absolute() else ROOT / args.base
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    audit_out = args.audit_out if args.audit_out.is_absolute() else ROOT / args.audit_out

    if not base_path.exists():
        raise SystemExit(f"Missing base split: {base_path}")

    base = json.loads(base_path.read_text(encoding="utf-8"))
    if not base.get("train"):
        raise SystemExit(f"Base split has empty train: {base_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict] = []

    for frac in args.fracs:
        if not (0.0 < float(frac) <= 1.0):
            raise SystemExit(f"frac must be in (0, 1], got {frac}")
        pct = _pct_tag(float(frac))
        ssl, coupled, meta, checks = build_for_frac(base, base_path, float(frac), args.seed)

        ssl_path = out_dir / f"limited_label_f{pct}_ssl.json"
        coupled_path = out_dir / f"limited_label_f{pct}_coupled.json"
        meta_path = out_dir / f"limited_label_f{pct}_meta.json"
        ssl_path.write_text(json.dumps(ssl, indent=2) + "\n", encoding="utf-8")
        coupled_path.write_text(json.dumps(coupled, indent=2) + "\n", encoding="utf-8")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        print(
            f"[f{pct}] coupled={checks['n_labeled']} unlabeled={checks['n_unlabeled']} "
            f"→ {ssl_path.name}, {coupled_path.name}, {meta_path.name}"
        )
        if not checks["intersection_empty"] or not checks["union_covers_base"]:
            raise SystemExit(f"Integrity failed for frac={frac}: {checks}")

        audit_rows.append(
            {
                "frac": float(frac),
                "pct": pct,
                "val_n": len(ssl["val"]),
                **checks,
            }
        )

    write_audit(audit_out, audit_rows, args.seed, base_path)
    print(f"Wrote audit {audit_out}")


if __name__ == "__main__":
    main()

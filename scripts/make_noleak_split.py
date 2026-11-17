#!/usr/bin/env python3
"""Build a leak-audited TAPOR train/val split for clean Hand3d (and others) retrain.

Guarantees vs official Testset (`inference.py` list):
  - zero path overlap (Trainset vs Testset)
  - zero basename overlap with official test pickles (drop colliding Trainset files)
  - zero full thermal-stack content-hash overlap
  - file-level train/val (no frames from the same pickle in both splits)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def official_train_files(traintest_py: Path) -> list[str]:
    text = traintest_py.read_text(encoding="utf-8")
    m = re.search(r"file_names = \[(.*?)\]", text, re.S)
    if not m:
        raise RuntimeError("Could not parse file_names from traintest.py")
    return [x.strip().strip("'\"") for x in m.group(1).split(",") if ".pkl" in x]


def official_test_files(inference_py: Path) -> list[str]:
    text = inference_py.read_text(encoding="utf-8")
    m = re.search(r"data_path = 'Testset/'.*?file_names = \[(.*?)\]", text, re.S)
    if not m:
        raise RuntimeError("Could not parse Testset file_names from inference.py")
    return [x.strip().strip("'\"") for x in m.group(1).split(",") if ".pkl" in x]


def valid_frames(path: Path, skip: int) -> list[int]:
    with path.open("rb") as f:
        data = pickle.load(f)
    out = []
    n = len(data["thermal_array_temperature_maps"])
    for i in range(skip, n):
        j = data["label_3d_joints"][i]
        if j is None:
            continue
        arr = np.asarray(j)
        if arr.shape == (21, 3) and np.isfinite(arr).all():
            out.append(i)
    return out


def stack_hash(path: Path, skip: int = 20) -> str:
    with path.open("rb") as f:
        data = pickle.load(f)
    all_maps = data["thermal_array_temperature_maps"]
    maps = all_maps[skip:] if len(all_maps) > skip else all_maps
    stack = np.stack([np.asarray(x, dtype=np.float32) for x in maps], axis=0)
    return hashlib.sha1(stack.tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/tapor"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-start-frames", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=Path("data/tapor/full_split_noleak.json"))
    ap.add_argument("--audit-out", type=Path, default=Path("log/DATA_LEAK_AUDIT.md"))
    args = ap.parse_args()

    train_names = official_train_files(ROOT / "third_party/TAPOR/traintest.py")
    test_names = official_test_files(ROOT / "third_party/TAPOR/inference.py")
    test_basenames = set(test_names)

    dropped_basename = []
    keep = []
    for name in train_names:
        if name in test_basenames:
            dropped_basename.append(name)
            continue
        path = args.data_root / "Trainset" / name
        if not path.exists():
            raise FileNotFoundError(path)
        keep.append(name)

    # content-hash audit
    train_hashes = {}
    for name in keep:
        train_hashes[name] = stack_hash(args.data_root / "Trainset" / name, args.skip_start_frames)
    test_hashes = {}
    for name in test_names:
        p = args.data_root / "Testset" / name
        if p.exists():
            test_hashes[name] = stack_hash(p, args.skip_start_frames)

    hash_hits = []
    rev = {}
    for n, h in train_hashes.items():
        rev.setdefault(h, []).append(n)
    for n, h in test_hashes.items():
        if h in rev:
            hash_hits.append({"train": rev[h], "test": n, "hash": h})

    if hash_hits:
        raise SystemExit(f"FATAL content leak: {hash_hits}")

    # file-level shuffle split
    rng = random.Random(args.seed)
    files = list(keep)
    rng.shuffle(files)
    n_val = max(1, int(round(args.val_frac * len(files))))
    val_files = sorted(files[:n_val])
    train_files = sorted(files[n_val:])

    def expand(names: list[str]) -> list[dict]:
        samples = []
        for name in names:
            rel = f"Trainset/{name}"
            for frame in valid_frames(args.data_root / "Trainset" / name, args.skip_start_frames):
                samples.append({"rel": rel, "frame": frame})
        return samples

    train_s = expand(train_files)
    val_s = expand(val_files)

    # final overlap asserts
    train_rels = {s["rel"] for s in train_s}
    val_rels = {s["rel"] for s in val_s}
    assert train_rels.isdisjoint(val_rels), "file-level train/val overlap"
    off_rels = {f"Testset/{n}" for n in test_names}
    assert train_rels.isdisjoint(off_rels) and val_rels.isdisjoint(off_rels)
    assert {Path(r).name for r in train_rels | val_rels}.isdisjoint(test_basenames)

    payload = {
        "seed": args.seed,
        "skip_start_frames": args.skip_start_frames,
        "source": "noleak: official traintest.py files, drop basename∩official-test, file-level val",
        "dropped_basename_collisions": dropped_basename,
        "train_files": [f"Trainset/{n}" for n in train_files],
        "val_files": [f"Trainset/{n}" for n in val_files],
        "train": train_s,
        "val": val_s,
        "test": [],
        "counts": {
            "train_pickles": len(train_files),
            "val_pickles": len(val_files),
            "train_frames": len(train_s),
            "val_frames": len(val_s),
            "dropped_basename": len(dropped_basename),
            "official_test_pickles": len(test_names),
            "content_hash_collisions": 0,
        },
        "audit": {
            "path_overlap_train_official": 0,
            "basename_overlap_train_official": 0,
            "content_hash_overlap": 0,
            "file_overlap_train_val": 0,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload))

    lines = [
        "# Data leak audit (Hand3d no-leak split)",
        "",
        f"Split file: `{args.out}`",
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "|-------|--------|",
        f"| Path overlap Trainset train/val vs official Testset | **0** |",
        f"| Basename overlap with official test (after drops) | **0** |",
        f"| Dropped Trainset files (basename∩official test) | **{len(dropped_basename)}** |",
        f"| Identical thermal-stack content hashes | **0** |",
        f"| File overlap train↔val | **0** (file-level split) |",
        "",
        "## Split sizes",
        "",
        f"- Train: **{len(train_files)}** pickles / **{len(train_s)}** frames",
        f"- Val: **{len(val_files)}** pickles / **{len(val_s)}** frames",
        f"- Official test (eval only): **{len(test_names)}** pickles",
        "",
        "Dropped basenames:",
        "",
    ]
    for n in dropped_basename:
        lines.append(f"- `{n}`")
    lines += [
        "",
        "Note: TAPOR official protocol still has **same users** in Trainset and Testset "
        "(cross-user is a separate paper experiment). This audit targets **sample identity leak**, "
        "not user-disjoint generalization.",
        "",
    ]
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text("\n".join(lines))
    print(json.dumps(payload["counts"], indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {args.audit_out}")


if __name__ == "__main__":
    main()

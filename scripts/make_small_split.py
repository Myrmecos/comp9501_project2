#!/usr/bin/env python3
"""Build a tiny reproducible TAPOR train/val split for smoke training."""
from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np


def list_pkls(directory: Path):
    return sorted(p for p in directory.glob("*.pkl") if not p.name.startswith("._"))


def count_valid_joints(path: Path, skip: int = 20) -> int:
    with path.open("rb") as f:
        data = pickle.load(f)
    joints = data["label_3d_joints"]
    n = 0
    for item in joints[skip:]:
        if item is None:
            continue
        arr = np.asarray(item)
        if arr.shape == (21, 3) and np.isfinite(arr).all():
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/tapor"))
    parser.add_argument("--out", type=Path, default=Path("data/tapor/small_split.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-n", type=int, default=12)
    parser.add_argument("--val-n", type=int, default=4)
    parser.add_argument("--skip", type=int, default=20)
    args = parser.parse_args()

    train_dir = args.data_root / "Trainset"
    test_dir = args.data_root / "Testset"
    train_files = list_pkls(train_dir)
    test_files = list_pkls(test_dir)
    rng = random.Random(args.seed)

    # Prefer files with enough valid joints
    scored_train = [(count_valid_joints(p, args.skip), p) for p in train_files]
    scored_train = [(s, p) for s, p in scored_train if s >= 30]
    scored_train.sort(key=lambda x: (-x[0], x[1].name))
    pool_train = [p for _, p in scored_train] or train_files
    rng.shuffle(pool_train)
    train_sel = pool_train[: args.train_n]

    scored_test = [(count_valid_joints(p, args.skip), p) for p in test_files]
    scored_test = [(s, p) for s, p in scored_test if s >= 30]
    scored_test.sort(key=lambda x: (-x[0], x[1].name))
    pool_val = [p for _, p in scored_test] or test_files
    rng.shuffle(pool_val)
    val_sel = pool_val[: args.val_n]

    payload = {
        "seed": args.seed,
        "skip_start_frames": args.skip,
        "train": [str(p.relative_to(args.data_root)) for p in train_sel],
        "val": [str(p.relative_to(args.data_root)) for p in val_sel],
        "train_valid_joint_frames": {p.name: count_valid_joints(p, args.skip) for p in train_sel},
        "val_valid_joint_frames": {p.name: count_valid_joints(p, args.skip) for p in val_sel},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out}")
    print("train:", [p.name for p in train_sel])
    print("val:", [p.name for p in val_sel])


if __name__ == "__main__":
    main()

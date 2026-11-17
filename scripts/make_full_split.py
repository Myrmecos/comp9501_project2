#!/usr/bin/env python3
"""Build full TAPOR train/val split matching official traintest.py file list + 80/10/10."""
from __future__ import annotations

import argparse
import json
import pickle
import random
import re
from pathlib import Path

import numpy as np


def official_train_files(traintest_py: Path) -> list[str]:
    text = traintest_py.read_text(encoding="utf-8")
    m = re.search(r"file_names = \[(.*?)\]", text, re.S)
    if not m:
        raise RuntimeError("Could not parse file_names from traintest.py")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/tapor"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-start-frames", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("data/tapor/full_split.json"))
    args = ap.parse_args()

    names = official_train_files(Path("third_party/TAPOR/traintest.py"))
    train_dir = args.data_root / "Trainset"
    samples = []
    for name in names:
        path = train_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        for frame in valid_frames(path, args.skip_start_frames):
            samples.append({"rel": f"Trainset/{name}", "frame": frame})

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    n = len(samples)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_s = samples[:n_train]
    val_s = samples[n_train : n_train + n_val]
    holdout_s = samples[n_train + n_val :]

    test_dir = args.data_root / "Testset"
    test_files = sorted(p.name for p in test_dir.glob("*.pkl")) if test_dir.exists() else []

    payload = {
        "seed": args.seed,
        "skip_start_frames": args.skip_start_frames,
        "source": "official TAPOR traintest.py file_names + 80/10/10 frame split",
        "train_files": [f"Trainset/{n}" for n in names],
        "train": train_s,
        "val": val_s,
        "holdout_from_trainset": holdout_s,
        "test_files": [f"Testset/{n}" for n in test_files],
        "counts": {
            "train_frames": len(train_s),
            "val_frames": len(val_s),
            "holdout_frames": len(holdout_s),
            "train_pickles": len(names),
            "test_pickles": len(test_files),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["counts"], indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build stratified train/val/test split for TAPOR gesture pickles (seed=0)."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import GESTURE_LABEL_FROM_NAME, parse_gesture_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/tapor"))
    ap.add_argument("--gesture-dir", type=Path, default=Path("data/tapor/gesture"))
    ap.add_argument("--out", type=Path, default=Path("data/tapor/gesture_split.json"))
    ap.add_argument("--skip-start-frames", type=int, default=20)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    files = sorted(args.gesture_dir.glob("*.pkl"))
    if not files:
        raise SystemExit(f"No pickles in {args.gesture_dir}")

    by_label: dict[int, list[dict]] = defaultdict(list)
    for path in files:
        label = parse_gesture_label(path)
        rel = str(path.relative_to(args.data_root))
        with path.open("rb") as f:
            data = pickle.load(f)
        n = len(data["thermal_array_temperature_maps"])
        for frame in range(args.skip_start_frames, n):
            by_label[label].append({"rel": rel, "frame": frame, "label": label})

    train, val, test = [], [], []
    counts = {}
    for label, items in sorted(by_label.items()):
        idx = np.arange(len(items))
        rng.shuffle(idx)
        n = len(items)
        n_train = int(round(args.train_frac * n))
        n_val = int(round(args.val_frac * n))
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            n_train = max(1, n_train - 1)
        tr = idx[:n_train]
        va = idx[n_train : n_train + n_val]
        te = idx[n_train + n_val :]
        train.extend(items[i] for i in tr)
        val.extend(items[i] for i in va)
        test.extend(items[i] for i in te)
        counts[str(label)] = {"n": n, "train": len(tr), "val": len(va), "test": len(te)}

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    payload = {
        "seed": args.seed,
        "skip_start_frames": args.skip_start_frames,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "classes": {str(k): v for k, v in GESTURE_LABEL_FROM_NAME.items()},
        "counts": {
            "files": len(files),
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "per_class": counts,
        },
        "train": train,
        "val": val,
        "test": test,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload))
    print(json.dumps(payload["counts"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

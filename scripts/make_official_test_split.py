#!/usr/bin/env python3
"""Build official TAPOR Testset split from inference.py file list."""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np


def official_test_files(inference_py: Path) -> list[str]:
    text = inference_py.read_text(encoding="utf-8")
    # first file_names = [ ... ] after data_path = 'Testset/'
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/tapor"))
    ap.add_argument("--skip-start-frames", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("data/tapor/official_test_split.json"))
    args = ap.parse_args()

    names = official_test_files(Path("third_party/TAPOR/inference.py"))
    test_dir = args.data_root / "Testset"
    samples = []
    missing = []
    empty = []
    for name in names:
        path = test_dir / name
        if not path.exists():
            missing.append(name)
            continue
        frames = valid_frames(path, args.skip_start_frames)
        if not frames:
            empty.append(name)
            continue
        for frame in frames:
            samples.append({"rel": f"Testset/{name}", "frame": frame})

    payload = {
        "seed": 0,
        "skip_start_frames": args.skip_start_frames,
        "source": "official TAPOR inference.py Testset file_names",
        "train": [],  # unused; eval uses split=val or we map test→val
        "val": samples,  # reuse dataset API with split='val'
        "test": samples,
        "missing_files": missing,
        "empty_files": empty,
        "counts": {
            "listed_pickles": len(names),
            "found_pickles": len(names) - len(missing),
            "test_frames": len(samples),
            "missing": len(missing),
            "empty": len(empty),
        },
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["counts"], indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

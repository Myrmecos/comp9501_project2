#!/usr/bin/env python3
"""Inspect trusted TAPOR pickle data without modifying it or running a model."""
import argparse
import ast
import datetime as dt
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REQUIRED_KEYS = (
    "thermal_array_temperature_maps",
    "thermal_array_ambient_temperatures",
    "label_depth_maps",
    "label_2d_joints",
    "label_3d_joints",
    "label_hand_depth",
)


def official_lists(tapor_root: Path):
    """Parse the upstream literal filename lists instead of maintaining copies."""
    def literals_after_assignment(path: Path, variable: str):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        values = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == variable for target in node.targets
            ) and isinstance(node.value, (ast.List, ast.Tuple)):
                literal = ast.literal_eval(node.value)
                if literal and all(isinstance(item, str) for item in literal):
                    values.append(literal)
        if not values:
            raise RuntimeError(f"No literal {variable} list found in {path}")
        return max(values, key=len)

    return (
        literals_after_assignment(tapor_root / "traintest.py", "file_names"),
        literals_after_assignment(tapor_root / "inference.py", "file_names"),
    )


def value_items(value):
    """Return per-frame items without coercing an entire potentially ragged sequence."""
    if isinstance(value, (list, tuple)):
        return value
    if isinstance(value, np.ndarray) and value.ndim > 0:
        return value
    return None


def frame_array(value):
    """Convert one frame only; ragged annotations must not poison a whole sequence."""
    try:
        return np.asarray(value), None
    except (TypeError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def sequence_stats(value, sample_limit):
    """Describe a sequence frame-by-frame and retain finite numeric statistics."""
    items = value_items(value)
    if items is None:
        array, error = frame_array(value)
        return {
            "container": "scalar",
            "frame_count": None,
            "frame_shapes": {str(list(array.shape)) if array is not None else "<unconvertible>": 1},
            "frame_dtypes": {str(array.dtype) if array is not None else "<unknown>": 1},
            "ragged_frame_count": int(error is not None),
            "conversion_errors": [error] if error else [],
        }

    shape_counts = Counter()
    dtype_counts = Counter()
    errors = []
    samples = []
    numeric_frames = 0
    for index, item in enumerate(items):
        array, error = frame_array(item)
        if error:
            shape_counts["<unconvertible>"] += 1
            errors.append({"frame": index, "error": error})
            continue
        shape_counts[str(list(array.shape))] += 1
        dtype_counts[str(array.dtype)] += 1
        if np.issubdtype(array.dtype, np.number):
            numeric_frames += 1
            flat = array.reshape(-1)
            remaining = max(sample_limit - len(samples), 0)
            if remaining:
                samples.extend(flat[:remaining].astype(np.float64, copy=False).tolist())

    result = {
        "container": type(value).__name__,
        "frame_count": len(items),
        "frame_shapes": dict(shape_counts),
        "frame_dtypes": dict(dtype_counts),
        "numeric_frame_count": numeric_frames,
        "ragged_frame_count": len(errors),
        "conversion_errors": errors[:10],
        "statistics_sampled": len(samples) >= sample_limit,
    }
    if samples:
        sample_array = np.asarray(samples, dtype=np.float64)
        finite = np.isfinite(sample_array)
        result.update(
            finite_fraction=float(finite.mean()),
            min=float(np.min(sample_array[finite])) if finite.any() else None,
            max=float(np.max(sample_array[finite])) if finite.any() else None,
            mean=float(np.mean(sample_array[finite])) if finite.any() else None,
        )
    return result


def labels_summary(labels, expected_frames, expected_joint_shape=(21, 3)):
    valid = 0
    empty = 0
    malformed = 0
    conversion_errors = []
    items = value_items(labels)
    if items is None:
        return {
            "frames_in_label": None,
            "expected_frames": expected_frames,
            "valid_21x3_finite": 0,
            "empty_or_missing_annotation": 0,
            "malformed_nonempty_annotation": 1,
            "ragged_annotation_entries": 0,
            "frame_count_matches_thermal": False,
            "error": "Label container is not a sequence.",
        }
    total = len(items)
    for index, label in enumerate(items):
        if label is None:
            empty += 1
            continue
        arr, error = frame_array(label)
        if error:
            malformed += 1
            conversion_errors.append({"frame": index, "error": error})
        elif arr.size == 0:
            empty += 1
        elif tuple(arr.shape) != expected_joint_shape:
            malformed += 1
        else:
            try:
                if np.isfinite(arr).all():
                    valid += 1
                else:
                    malformed += 1
            except TypeError as exc:
                malformed += 1
                conversion_errors.append({"frame": index, "error": f"TypeError: {exc}"})
    return {
        "frames_in_label": total,
        "expected_frames": expected_frames,
        "valid_21x3_finite": valid,
        "empty_or_missing_annotation": empty,
        "malformed_nonempty_annotation": malformed,
        "ragged_annotation_entries": len(conversion_errors),
        "conversion_errors": conversion_errors[:10],
        "frame_count_matches_thermal": total == expected_frames,
    }


def required_sequence_issue(name, value, expected_frames, expected_ndim=None):
    """Check the fields consumed by official data_loader without whole-sequence coercion."""
    items = value_items(value)
    if items is None:
        return f"{name} is not a per-frame sequence."
    if len(items) != expected_frames:
        return f"{name} has {len(items)} frames; thermal sequence has {expected_frames}."
    shapes = set()
    for index, item in enumerate(items):
        array, error = frame_array(item)
        if error:
            return f"{name}[{index}] cannot be converted to an array: {error}"
        if not np.issubdtype(array.dtype, np.number):
            return f"{name}[{index}] has non-numeric dtype {array.dtype}."
        if expected_ndim is not None and array.ndim != expected_ndim:
            return f"{name}[{index}] has ndim {array.ndim}; expected {expected_ndim}."
        if not np.isfinite(array).all():
            return f"{name}[{index}] contains non-finite values."
        shapes.add(tuple(array.shape))
    if len(shapes) > 1:
        return f"{name} has inconsistent per-frame shapes {sorted(shapes)}; official dataset.py uses np.array on this sequence."
    return None


def inspect_pickle(path: Path, sample_limit: int):
    outcome = {"path": str(path), "name": path.name}
    try:
        # Pickle is executable by design. Only inspect data obtained from the trusted team source.
        with path.open("rb") as handle:
            data = pickle.load(handle)
    except Exception as exc:
        outcome.update(status="unreadable", error=f"{type(exc).__name__}: {exc}")
        return outcome
    if not isinstance(data, dict):
        outcome.update(status="invalid", error=f"Expected dict, got {type(data).__name__}")
        return outcome
    missing = [key for key in REQUIRED_KEYS if key not in data]
    outcome["missing_keys"] = missing
    if missing:
        outcome["status"] = "invalid"
        return outcome
    try:
        thermal_items = value_items(data["thermal_array_temperature_maps"])
        if thermal_items is None:
            outcome.update(status="invalid", error="thermal_array_temperature_maps is not a per-frame sequence.")
            return outcome
        frames = int(len(thermal_items))
        outcome["frames"] = frames
        outcome["keys"] = {key: sequence_stats(data[key], sample_limit) for key in REQUIRED_KEYS}
        outcome["label_2d_summary"] = labels_summary(data["label_2d_joints"], frames)
        outcome["label_3d_summary"] = labels_summary(data["label_3d_joints"], frames)
        structural_issues = [
            issue for issue in (
                required_sequence_issue("thermal_array_temperature_maps", data["thermal_array_temperature_maps"], frames, expected_ndim=2),
                required_sequence_issue("thermal_array_ambient_temperatures", data["thermal_array_ambient_temperatures"], frames),
                required_sequence_issue("label_depth_maps", data["label_depth_maps"], frames, expected_ndim=2),
            ) if issue
        ]
        for key in ("label_2d_joints", "label_3d_joints", "label_hand_depth"):
            items = value_items(data[key])
            if items is None or len(items) != frames:
                structural_issues.append(f"{key} does not have one entry per thermal frame.")
        outcome["structural_issues"] = structural_issues
        outcome["annotation_warnings"] = {
            "label_2d": outcome["label_2d_summary"],
            "label_3d": outcome["label_3d_summary"],
        }
        outcome["depth_label_note"] = (
            "Depth-map values are reported above but are not TAPOR's primary supervision; "
            "traintest.py supervises label_3d_joints."
        )
        outcome["official_loader_compatible"] = frames > 20 and not structural_issues
        outcome["status"] = "ok" if outcome["official_loader_compatible"] else "invalid"
        if frames <= 20:
            outcome["error"] = "Official dataset.py discards the first 20 frames; sequence has no usable frame."
        elif structural_issues:
            outcome["error"] = " ; ".join(structural_issues)
    except Exception as exc:
        outcome.update(status="invalid", error=f"inspection error: {type(exc).__name__}: {exc}")
    return outcome


def is_macos_metadata(path: Path):
    return (
        "__MACOSX" in path.parts
        or ".AppleDouble" in path.parts
        or path.name.startswith("._")
        or path.name in {".DS_Store", "Icon\r"}
    )


def locate(root: Path, expected):
    by_name = {}
    for path in root.rglob("*.pkl"):
        if is_macos_metadata(path):
            continue
        by_name.setdefault(path.name, []).append(path)
    expected_set = set(expected)
    matched = {name: [str(path) for path in paths] for name, paths in by_name.items() if name in expected_set}
    return by_name, matched


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tapor-root", type=Path, default=Path("/workspace/TAPOR"))
    parser.add_argument("--data-root", type=Path, default=Path("/root/sj-tmp/tapor_data"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--sample-values", type=int, default=10000)
    parser.add_argument("--max-files", type=int, default=0, help="0 means inspect every discovered pickle")
    parser.add_argument("--require-complete", action="store_true", help="exit nonzero when official pose files are missing/invalid")
    args = parser.parse_args()
    if not (args.tapor_root / "traintest.py").is_file() or not (args.tapor_root / "inference.py").is_file():
        raise SystemExit(f"Not a TAPOR source root: {args.tapor_root}")
    if not args.data_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {args.data_root}")
    train_expected, test_expected = official_lists(args.tapor_root)
    all_expected = train_expected + test_expected
    by_name, matched = locate(args.data_root, all_expected)
    train_dir = args.data_root / "Trainset"
    test_dir = args.data_root / "Testset"
    train_by_name, _ = locate(train_dir, train_expected) if train_dir.is_dir() else ({}, {})
    test_by_name, _ = locate(test_dir, test_expected) if test_dir.is_dir() else ({}, {})
    train_missing = sorted(set(train_expected) - set(train_by_name))
    test_missing = sorted(set(test_expected) - set(test_by_name))
    duplicates = {name: [str(path) for path in paths] for name, paths in by_name.items() if len(paths) > 1}
    overlap = sorted(set(train_expected) & set(test_expected))
    verified_content_overlap = []
    overlap_hash_errors = []
    for name in overlap:
        if name not in train_by_name or name not in test_by_name:
            continue
        try:
            train_hash = sha256(train_by_name[name][0])
            test_hash = sha256(test_by_name[name][0])
            if train_hash == test_hash:
                verified_content_overlap.append({"name": name, "sha256": train_hash})
        except OSError as exc:
            overlap_hash_errors.append({"name": name, "error": str(exc)})
    targets = sorted((path for paths in by_name.values() for path in paths), key=lambda item: str(item))
    if args.max_files:
        targets = targets[: args.max_files]
    reports = [inspect_pickle(path, args.sample_values) for path in targets]
    invalid = [item["path"] for item in reports if item["status"] != "ok"]
    invalid_required = [
        item["path"] for item in reports
        if not item.get("official_loader_compatible", False) and item["name"] in set(all_expected)
    ]
    report = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "Static inspection only; no model training or evaluation was performed.",
        "data_root": str(args.data_root.resolve()),
        "tapor_root": str(args.tapor_root.resolve()),
        "official_expected_counts": {"train": len(train_expected), "test": len(test_expected)},
        "discovered_pickle_count": sum(len(paths) for paths in by_name.values()),
        "expected_split_directories_present": {"Trainset": train_dir.is_dir(), "Testset": test_dir.is_dir()},
        "official_train_missing": train_missing,
        "official_test_missing": test_missing,
        "official_train_test_filename_overlap": overlap,
        "verified_train_test_content_overlap": verified_content_overlap,
        "overlap_hash_errors": overlap_hash_errors,
        "duplicate_basenames": duplicates,
        "official_matched_paths": matched,
        "inspected_count": len(reports),
        "invalid_or_unreadable": invalid,
        "invalid_or_unreadable_official_pose_files": invalid_required,
        "files": reports,
    }
    report_path = args.report or args.data_root / "dataset_inspection_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False, default=str), encoding="utf-8")
    print(f"Wrote report: {report_path}")
    print(f"Discovered {report['discovered_pickle_count']} pickle files; inspected {len(reports)}.")
    print(f"Missing official pose train/test files: {len(train_missing)}/{len(test_missing)}")
    print(f"Invalid or unreadable inspected files: {len(invalid)}")
    if overlap:
        print("WARNING: official source lists share basenames; hashes were checked only when both split directories exist:", ", ".join(overlap))
    if verified_content_overlap:
        print("ERROR: byte-identical train/test data found:", ", ".join(item["name"] for item in verified_content_overlap))
    if args.require_complete and (not train_dir.is_dir() or not test_dir.is_dir() or train_missing or test_missing or invalid_required or verified_content_overlap):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Read-only TAPOR result_dict audit.

This tool deliberately mirrors external/TAPOR/utils.py's MPJPE/PCK definitions
and records alternate aggregations without changing any source result artifact.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def load_result(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    required = {"infernece_file_name", "failed_file_names", "predicts", "labels"}
    missing = required.difference(value)
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")
    lengths = {key: len(value[key]) for key in ("infernece_file_name", "predicts", "labels")}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{path}: inconsistent list lengths {lengths}")
    return value


def metric_summary(predicts: list[Any], labels: list[Any]) -> dict[str, Any]:
    """Return pooled and alternative aggregation metrics in metres/mm.

    Pooled MPJPE/PCK reproduce calculate_mpjpe/calculate_pck in utils.py:
    norm over XYZ per joint, followed by mean/count over every frame-joint pair.
    """
    file_metrics: list[dict[str, Any]] = []
    all_errors: list[np.ndarray] = []
    all_root_errors: list[np.ndarray] = []
    all_shifted_errors: list[np.ndarray] = []
    per_joint_errors: list[list[np.ndarray]] = [[] for _ in range(21)]
    invalid_shape_files: list[dict[str, str]] = []

    for file_index, (prediction, label) in enumerate(zip(predicts, labels)):
        prediction = np.asarray(prediction)
        label = np.asarray(label)
        if prediction.shape != label.shape or prediction.ndim != 3 or prediction.shape[1:] != (21, 3):
            invalid_shape_files.append(
                {"index": file_index, "prediction_shape": str(prediction.shape), "label_shape": str(label.shape)}
            )
            continue
        if not np.isfinite(prediction).all() or not np.isfinite(label).all():
            invalid_shape_files.append({"index": file_index, "reason": "non-finite prediction or label"})
            continue

        errors = np.linalg.norm(prediction - label, axis=-1)
        root_delta = label[:, 0, :] - prediction[:, 0, :]
        root_errors = np.linalg.norm(root_delta, axis=-1)
        shifted_errors = np.linalg.norm(prediction + root_delta[:, None, :] - label, axis=-1)
        all_errors.append(errors)
        all_root_errors.append(root_errors)
        all_shifted_errors.append(shifted_errors)
        for joint in range(21):
            per_joint_errors[joint].append(errors[:, joint])
        file_metrics.append(
            {
                "file_index": file_index,
                "frames": int(prediction.shape[0]),
                "mpjpe_mm": float(errors.mean() * 1000),
                "pck_4cm_percent": float((errors < 0.04).mean() * 100),
                "root_drift_mm": float(root_errors.mean() * 1000),
                "root_aligned_mpjpe_mm": float(shifted_errors.mean() * 1000),
                "root_aligned_pck_4cm_percent": float((shifted_errors < 0.04).mean() * 100),
            }
        )

    if not all_errors:
        raise ValueError("No valid (frames, 21, 3) prediction/label arrays")
    errors = np.concatenate(all_errors, axis=0)
    root_errors = np.concatenate(all_root_errors, axis=0)
    shifted_errors = np.concatenate(all_shifted_errors, axis=0)
    frame_mpjpe = errors.mean(axis=1)
    frame_pck = (errors < 0.04).mean(axis=1)
    joint_mpjpe = [float(np.concatenate(items).mean() * 1000) for items in per_joint_errors]
    joint_pck = [float((np.concatenate(items) < 0.04).mean() * 100) for items in per_joint_errors]
    file_mpjpe = np.array([item["mpjpe_mm"] for item in file_metrics])
    file_pck = np.array([item["pck_4cm_percent"] for item in file_metrics])

    return {
        "valid_files": len(file_metrics),
        "invalid_shape_or_nonfinite_files": invalid_shape_files,
        "frames": int(errors.shape[0]),
        "joint_instances": int(errors.size),
        "all_labels_are_finite": True,
        "zero_3d_label_frames": int(np.all(np.isclose(np.concatenate(labels, axis=0), 0.0), axis=(1, 2)).sum()),
        "official_pooled": {
            "mpjpe_mm": float(errors.mean() * 1000),
            "pck_4cm_percent": float((errors < 0.04).mean() * 100),
            "root_drift_mm": float(root_errors.mean() * 1000),
            "root_aligned_mpjpe_mm_diagnostic_only": float(shifted_errors.mean() * 1000),
            "root_aligned_pck_4cm_percent_diagnostic_only": float((shifted_errors < 0.04).mean() * 100),
        },
        "frame_mean": {
            "mpjpe_mm": float(frame_mpjpe.mean() * 1000),
            "pck_4cm_percent": float(frame_pck.mean() * 100),
        },
        "joint_mean": {
            "mean_of_21_joint_mpjpe_mm": float(np.mean(joint_mpjpe)),
            "mean_of_21_joint_pck_4cm_percent": float(np.mean(joint_pck)),
            "per_joint_mpjpe_mm": joint_mpjpe,
            "per_joint_pck_4cm_percent": joint_pck,
        },
        "file_mean_equal_weight": {
            "mpjpe_mm": float(file_mpjpe.mean()),
            "pck_4cm_percent": float(file_pck.mean()),
            "root_drift_mm": float(np.mean([item["root_drift_mm"] for item in file_metrics])),
            "root_aligned_mpjpe_mm_diagnostic_only": float(
                np.mean([item["root_aligned_mpjpe_mm"] for item in file_metrics])
            ),
        },
        "frame_count_distribution": {str(key): value for key, value in sorted(Counter(item["frames"] for item in file_metrics).items())},
        "per_file": file_metrics,
    }


def compact_author_results(log_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(log_root.rglob("*.pkl")):
        try:
            result = load_result(path)
            summary = metric_summary(result["predicts"], result["labels"])
            rows.append(
                {
                    "path": str(path),
                    "files": summary["valid_files"],
                    "frames": summary["frames"],
                    "failed_files": len(result["failed_file_names"]),
                    "mpjpe_mm": summary["official_pooled"]["mpjpe_mm"],
                    "pck_4cm_percent": summary["official_pooled"]["pck_4cm_percent"],
                    "file_mean_mpjpe_mm": summary["file_mean_equal_weight"]["mpjpe_mm"],
                }
            )
        except Exception as exc:  # Archive has heterogeneous, undocumented pickles.
            rows.append({"path": str(path), "unreadable_as_result_dict": f"{type(exc).__name__}: {exc}"})
    return rows


def labels_match(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    names_equal = left["infernece_file_name"] == right["infernece_file_name"]
    mismatch = []
    for index, (first, second) in enumerate(zip(left["labels"], right["labels"])):
        first, second = np.asarray(first), np.asarray(second)
        if first.shape != second.shape or not np.array_equal(first, second):
            mismatch.append(index)
    return {"file_names_equal": names_equal, "label_arrays_equal": not mismatch, "mismatching_label_indices": mismatch}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--retrained", type=Path, required=True)
    parser.add_argument("--official-logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pretrained = load_result(args.pretrained)
    retrained = load_result(args.retrained)
    payload = {
        "method": {
            "source": "external/TAPOR/utils.py: calculate_mpjpe (lines 103-111), calculate_pck (113-129), calculate_shift_error (81-100)",
            "units": "Input coordinates are retained in metres; report converts scalar distances to mm and PCK uses threshold 0.04 m.",
            "scope": "Read-only analysis of saved result_dict.pkl files. No model inference is run.",
        },
        "pretrained": metric_summary(pretrained["predicts"], pretrained["labels"]),
        "retrained": metric_summary(retrained["predicts"], retrained["labels"]),
        "cross_model_protocol": labels_match(pretrained, retrained),
        "official_log_archive": compact_author_results(args.official_logs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

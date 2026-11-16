#!/usr/bin/env python3
"""Read-only, per-file TAPOR A/B Testset comparison.

A is the current 207-file inference result.  B is an explicitly selected
historical author result_dict, not an inferred paper test split.  The script
never opens a raw dataset pkl and never changes input artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def load(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    required = {"infernece_file_name", "failed_file_names", "predicts", "labels"}
    if not required.issubset(data):
        raise ValueError(f"{path} is not a pose result_dict; missing {sorted(required - set(data))}")
    if not (len(data["infernece_file_name"]) == len(data["predicts"]) == len(data["labels"])):
        raise ValueError(f"{path} has inconsistent result_dict list lengths")
    return data


def metric(pred: np.ndarray, label: np.ndarray) -> dict[str, float | int]:
    pred, label = np.asarray(pred), np.asarray(label)
    if pred.shape != label.shape or pred.ndim != 3 or pred.shape[1:] != (21, 3):
        raise ValueError(f"expected matching (frames,21,3), got {pred.shape} and {label.shape}")
    error = np.linalg.norm(pred - label, axis=-1)
    return {
        "frames": int(pred.shape[0]),
        "mpjpe_mm": float(error.mean() * 1000),
        "pck_4cm_percent": float((error < 0.04).mean() * 100),
    }


def result_index(data: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    indexed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, pred, label in zip(data["infernece_file_name"], data["predicts"], data["labels"]):
        if name in indexed:
            raise ValueError(f"duplicate filename in result_dict: {name}")
        indexed[name] = (np.asarray(pred), np.asarray(label))
    return indexed


def pool(parts: Iterable[tuple[np.ndarray, np.ndarray]]) -> dict[str, float | int] | None:
    items = list(parts)
    if not items:
        return None
    return metric(np.concatenate([item[0] for item in items]), np.concatenate([item[1] for item in items]))


def frame_hashes(label: np.ndarray) -> list[bytes]:
    """Hash each 21x3 label with dtype/shape canonicalised for exact matching."""
    label = np.ascontiguousarray(np.asarray(label, dtype=np.float64))
    return [hashlib.sha256(frame.tobytes()).digest() for frame in label]


def common_frame_indices(a_label: np.ndarray, b_label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Labels are normally unique enough; Counter/deque preserves multiplicity if not.
    positions: dict[bytes, list[int]] = defaultdict(list)
    for index, digest in enumerate(frame_hashes(b_label)):
        positions[digest].append(index)
    offsets: dict[bytes, int] = defaultdict(int)
    a_indices, b_indices = [], []
    for index, digest in enumerate(frame_hashes(a_label)):
        offset = offsets[digest]
        if offset < len(positions[digest]):
            a_indices.append(index)
            b_indices.append(positions[digest][offset])
            offsets[digest] += 1
    return np.asarray(a_indices, dtype=int), np.asarray(b_indices, dtype=int)


def fields(name: str) -> dict[str, str]:
    stem = name[:-4] if name.endswith(".pkl") else name
    tokens = stem.split("_")
    prefix = tokens[0] if tokens else "unknown"
    participant = next((item for item in tokens if re.fullmatch(r"U(?:\d+|X)", item)), "unknown")
    hand = next((item for item in tokens if item in {"L", "R"}), "unknown")
    lower = stem.lower()
    if "cover" in lower or "thick" in lower or "medicalglove" in lower or "_glove" in lower or "_nail" in lower:
        category = "covering"
    elif "lightcond" in lower:
        category = "light_condition"
    elif "_dist_" in lower:
        category = "distance"
    elif "_temp_" in lower:
        category = "ambient_temperature"
    elif any(item in lower for item in ("_hot_", "_warm_", "_normal_", "_scold_", "_vcold_")):
        category = "hand_temperature"
    elif any(item in lower for item in ("_empty_", "_body_", "_laptop_", "_display_", "_hotbottle_", "_light_")):
        category = "ambient_interference"
    elif "_random_" in lower:
        category = "tracking"
    elif "_x_x_" in lower:
        category = "standard_x_x"
    else:
        category = "unclassified"
    return {"prefix": prefix, "participant": participant, "hand": hand, "category": category}


def short_experiment_name(path: Path) -> str:
    return str(path).replace("\\", "/")


def history_summary(log_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(log_root.rglob("*.pkl")):
        if "tapor" not in path.name.lower():
            continue
        try:
            item = load(path)
            indexed = result_index(item)
            totals = pool(indexed.values())
            assert totals
            rows.append({
                "path": str(path),
                "model_record_name": path.name,
                "successful_files": len(indexed),
                "failed_files": len(item["failed_file_names"]),
                **totals,
            })
        except Exception as exc:
            rows.append({"path": str(path), "model_record_name": path.name, "parse_error": f"{type(exc).__name__}: {exc}"})
    return rows


def update_group(groups: dict[tuple[str, str], list[dict[str, Any]]], dimension: str, value: str, row: dict[str, Any]) -> None:
    groups[(dimension, value)].append(row)


def group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for dim in ("category", "prefix", "participant", "hand"):
            update_group(groups, dim, row[dim], row)
    output = []
    for (dimension, value), entries in sorted(groups.items()):
        result: dict[str, Any] = {"dimension": dimension, "value": value, "files": len(entries)}
        for side in ("a", "b"):
            present = [entry for entry in entries if entry[f"{side}_present"]]
            result[f"{side}_files"] = len(present)
            result[f"{side}_frames"] = sum(entry[f"{side}_frames"] or 0 for entry in present)
            for score in ("mpjpe_mm", "pck_4cm_percent"):
                parts = [entry[f"{side}_pair"] for entry in present]
                totals = pool(parts)
                result[f"{side}_{score}"] = None if totals is None else totals[score]
        a_retrained_parts = [entry["a_retrained_pair"] for entry in entries if entry["a_present"]]
        a_retrained_totals = pool(a_retrained_parts)
        result["a_retrained_mpjpe_mm"] = None if a_retrained_totals is None else a_retrained_totals["mpjpe_mm"]
        result["a_retrained_pck_4cm_percent"] = None if a_retrained_totals is None else a_retrained_totals["pck_4cm_percent"]
        result["a_only_files"] = sum(entry["membership"] == "A_only" for entry in entries)
        result["b_only_files"] = sum(entry["membership"] == "B_only" for entry in entries)
        result["common_files"] = sum(entry["membership"] == "common" for entry in entries)
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-pretrained", type=Path, required=True)
    parser.add_argument("--a-retrained", type=Path, required=True)
    parser.add_argument("--b-history", type=Path, required=True)
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    a_pre = result_index(load(args.a_pretrained))
    a_re = result_index(load(args.a_retrained))
    b = result_index(load(args.b_history))
    if list(a_pre) != list(a_re):
        raise ValueError("A pretrained and retrained filename order differs")
    for name in a_pre:
        if not np.array_equal(a_pre[name][1], a_re[name][1]):
            raise ValueError(f"A pretrained/retrained labels differ for {name}")

    all_names = sorted(set(a_pre) | set(b))
    rows: list[dict[str, Any]] = []
    common_matched_pre: list[tuple[np.ndarray, np.ndarray]] = []
    common_matched_re: list[tuple[np.ndarray, np.ndarray]] = []
    common_matched_b: list[tuple[np.ndarray, np.ndarray]] = []
    common_file_a_pre: list[tuple[np.ndarray, np.ndarray]] = []
    common_file_a_re: list[tuple[np.ndarray, np.ndarray]] = []
    common_file_b: list[tuple[np.ndarray, np.ndarray]] = []

    for name in all_names:
        a_present, b_present = name in a_pre, name in b
        info = fields(name)
        row: dict[str, Any] = {"file_name": name, **info, "a_present": a_present, "b_present": b_present}
        row["membership"] = "common" if a_present and b_present else "A_only" if a_present else "B_only"
        for key in ("a", "b"):
            row[f"{key}_frames"] = None
            row[f"{key}_mpjpe_mm"] = None
            row[f"{key}_pck_4cm_percent"] = None
            row[f"{key}_pair"] = None
        row["a_retrained_mpjpe_mm"] = None
        row["a_retrained_pck_4cm_percent"] = None
        row["a_retrained_pair"] = None
        if a_present:
            row["a_pair"] = a_pre[name]
            row.update({f"a_{key}": value for key, value in metric(*a_pre[name]).items()})
            row["a_retrained_pair"] = a_re[name]
            re_score = metric(*a_re[name])
            row["a_retrained_mpjpe_mm"] = re_score["mpjpe_mm"]
            row["a_retrained_pck_4cm_percent"] = re_score["pck_4cm_percent"]
        if b_present:
            row["b_pair"] = b[name]
            row.update({f"b_{key}": value for key, value in metric(*b[name]).items()})
        if a_present and b_present:
            common_file_a_pre.append(a_pre[name])
            common_file_a_re.append(a_re[name])
            common_file_b.append(b[name])
            a_indices, b_indices = common_frame_indices(a_pre[name][1], b[name][1])
            row["exact_label_matched_frames"] = int(len(a_indices))
            row["labels_same_full_array"] = bool(np.array_equal(a_pre[name][1], b[name][1]))
            row["same_frame_count"] = bool(a_pre[name][1].shape[0] == b[name][1].shape[0])
            row["a_only_frames_by_label"] = int(a_pre[name][1].shape[0] - len(a_indices))
            row["b_only_frames_by_label"] = int(b[name][1].shape[0] - len(b_indices))
            if len(a_indices):
                common_matched_pre.append((a_pre[name][0][a_indices], a_pre[name][1][a_indices]))
                common_matched_re.append((a_re[name][0][a_indices], a_pre[name][1][a_indices]))
                common_matched_b.append((b[name][0][b_indices], b[name][1][b_indices]))
        else:
            row.update({"exact_label_matched_frames": 0, "labels_same_full_array": False, "same_frame_count": False,
                        "a_only_frames_by_label": row["a_frames"] or 0, "b_only_frames_by_label": row["b_frames"] or 0})
        rows.append(row)

    # Remove in-memory ndarrays before JSON/CSV serialization.
    csv_rows = [{key: value for key, value in row.items() if not key.endswith("_pair")} for row in rows]
    names_a, names_b = set(a_pre), set(b)
    memberships = Counter(row["membership"] for row in rows)
    frame_diff_files = [row for row in csv_rows if row["membership"] == "common" and not row["same_frame_count"]]
    exact_parts = {
        "a_pretrained": pool(common_matched_pre),
        "a_retrained": pool(common_matched_re),
        "b_historical": pool(common_matched_b),
    }
    nominal_common = {
        "a_pretrained": pool(common_file_a_pre),
        "a_retrained": pool(common_file_a_re),
        "b_historical": pool(common_file_b),
    }
    frame_accounting = {
        "a_total_frames": sum(row["a_frames"] or 0 for row in csv_rows),
        "b_total_frames": sum(row["b_frames"] or 0 for row in csv_rows),
        "difference_a_minus_b": sum(row["a_frames"] or 0 for row in csv_rows) - sum(row["b_frames"] or 0 for row in csv_rows),
        "a_only_file_frames": sum(row["a_frames"] or 0 for row in csv_rows if row["membership"] == "A_only"),
        "b_only_file_frames": sum(row["b_frames"] or 0 for row in csv_rows if row["membership"] == "B_only"),
        "a_extra_frames_within_common_by_label": sum(row["a_only_frames_by_label"] for row in csv_rows if row["membership"] == "common"),
        "b_extra_frames_within_common_by_label": sum(row["b_only_frames_by_label"] for row in csv_rows if row["membership"] == "common"),
        "exactly_matched_common_frames": sum(row["exact_label_matched_frames"] for row in csv_rows if row["membership"] == "common"),
    }
    output = {
        "artifact_paths": {
            "a_pretrained": str(args.a_pretrained), "a_retrained": str(args.a_retrained),
            "b_historical": str(args.b_history), "official_logs_root": str(args.logs_root),
        },
        "b_definition": "The 102-file LogInference TAPOR 77 record selected because it was closest to the paper metrics. It is not asserted to be the paper Figure 12(c) test set.",
        "history_tapor_experiments": history_summary(args.logs_root),
        "set_counts": {"A_files": len(names_a), "B_files": len(names_b), "intersection_files": len(names_a & names_b),
                       "A_only_files": len(names_a - names_b), "B_only_files": len(names_b - names_a), **memberships},
        "frame_accounting": frame_accounting,
        "common_file_label_validation": {
            "common_files": memberships["common"],
            "files_with_identical_label_arrays": sum(row["labels_same_full_array"] for row in csv_rows if row["membership"] == "common"),
            "files_with_different_retained_frame_counts": len(frame_diff_files),
            "files_with_any_exact_label_frame_match": sum(row["exact_label_matched_frames"] > 0 for row in csv_rows if row["membership"] == "common"),
        },
        "common_file_metrics_nominal_full_retained_arrays_not_strict_if_lengths_differ": nominal_common,
        "common_exact_label_frame_metrics_strictly_comparable": exact_parts,
        "files_with_different_frame_counts": frame_diff_files,
        "file_rows": csv_rows,
        "category_summary": group_summary(rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "testset_overlap_analysis.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "testset_file_comparison.csv", csv_rows)
    write_csv(args.output_dir / "testset_category_summary.csv", output["category_summary"])


if __name__ == "__main__":
    main()

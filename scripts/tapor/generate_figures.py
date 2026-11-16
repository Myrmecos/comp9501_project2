#!/usr/bin/env python3
"""Generate GitHub-ready TAPOR audit figures from local, read-only artifacts."""
from __future__ import annotations

import csv
import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path.cwd()
FIGURES = Path.cwd() / "figures"
PRETRAINED = Path.cwd() / "missing_pretrained_result_dict.pkl"
RETRAINED = Path.cwd() / "missing_retrained_result_dict.pkl"
EVENT: Optional[Path] = None

plt.rcParams.update({"font.size": 11, "axes.titlepad": 10, "figure.dpi": 140})


def load_result(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def concat(result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate(result["predicts"], axis=0), np.concatenate(result["labels"], axis=0)


def errors(result: dict[str, Any]) -> np.ndarray:
    pred, label = concat(result)
    return np.linalg.norm(pred - label, axis=-1)


def pck_curve(error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds_m = np.array([0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
    return thresholds_m * 100, np.array([(error < value).mean() * 100 for value in thresholds_m])


def per_file_mpjpe(result: dict[str, Any]) -> np.ndarray:
    return np.array([np.linalg.norm(np.asarray(pred) - np.asarray(label), axis=-1).mean() * 1000
                     for pred, label in zip(result["predicts"], result["labels"])])


def save(fig: plt.Figure, name: str, records: list[dict[str, Any]], description: str, sources: list[str], placement: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=220, bbox_inches="tight")
    plt.close(fig)
    records.append({"filename": f"figures/{name}", "description": description, "data_sources": sources,
                    "recommended_placement": placement})


def line_plot(x: np.ndarray, y: np.ndarray, title: str, ylabel: str, filename: str, records: list[dict[str, Any]], tag: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(x, y, label=tag)
    ax.set(xlabel="Epoch", ylabel=ylabel, title=title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    save(fig, filename, records, title, ["TensorBoard event: reproduction_results/reproduced_training/events.out.tfevents.*"], "Analysis section")


def bar_plot(labels: list[str], values: list[float], title: str, ylabel: str, filename: str, records: list[dict[str, Any]], sources: list[str], placement: str, rotation: int = 0) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    bars = ax.bar(labels, values)
    ax.set(title=title, ylabel=ylabel)
    ax.tick_params(axis="x", rotation=rotation)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    save(fig, filename, records, title, sources, placement)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    global ROOT, FIGURES, PRETRAINED, RETRAINED, EVENT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True,
                        help="Directory containing audit_evidence.json, inference_results/, and reproduction_results/.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for generated PNG files.")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Optional output manifest path (default: <artifact-root>/figure_manifest.json).")
    args = parser.parse_args()
    ROOT = args.artifact_root.resolve()
    FIGURES = args.output_dir.resolve()
    PRETRAINED = ROOT / "inference_results" / "tapor_type1_weights_tapor.pth_2026111695944_result_dict.pkl"
    RETRAINED = ROOT / "inference_results" / "tapor_type1_weights_1_20261116103244_tapor.pth_2026111616649_result_dict.pkl"
    events = sorted((ROOT / "reproduction_results" / "reproduced_training").glob("events.out.tfevents.*"))
    if not events:
        raise SystemExit("No TensorBoard event file under <artifact-root>/reproduction_results/reproduced_training/.")
    EVENT = events[0]
    for required in (PRETRAINED, RETRAINED, ROOT / "audit_evidence.json", ROOT / "testset_overlap_analysis.json", ROOT / "testset_category_summary.csv"):
        if not required.is_file():
            raise SystemExit(f"Missing required audit artifact: {required}")
    records: list[dict[str, Any]] = []
    pre = load_result(PRETRAINED)
    ours = load_result(RETRAINED)
    pre_error, ours_error = errors(pre), errors(ours)
    evidence = json.loads((ROOT / "audit_evidence.json").read_text())
    overlap = json.loads((ROOT / "testset_overlap_analysis.json").read_text())
    categories = read_csv(ROOT / "testset_category_summary.csv")

    ea = EventAccumulator(str(EVENT))
    ea.Reload()
    scalar = lambda tag: ea.Scalars(tag)
    for tag, title, ylabel, filename in [
        ("train loss (average)", "TAPOR Reproduction Training Loss", "Average training loss", "training_loss_curve.png"),
        ("validation mse loss (average)", "TAPOR Reproduction Validation MSE", "Validation MSE", "validation_mse_curve.png"),
        ("validation mae loss (average)", "TAPOR Reproduction Validation MAE", "Validation MAE", "validation_mae_curve.png"),
    ]:
        points = scalar(tag)
        line_plot(np.array([item.step for item in points]), np.array([item.value for item in points]), title, ylabel, filename, records, tag)

    thresholds_cm, pre_pck = pck_curve(pre_error)
    _, ours_pck = pck_curve(ours_error)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(thresholds_cm, pre_pck, marker="o", label="Official pretrained")
    ax.plot(thresholds_cm, ours_pck, marker="o", label="Our reproduction")
    ax.set(xlabel="PCK threshold (cm)", ylabel="PCK (%)", title="PCK Curve: Official Pretrained vs Our Reproduction")
    ax.grid(True, alpha=0.3); ax.legend()
    save(fig, "pck_curve_official_vs_ours.png", records, "Independent-Testset PCK curves computed from all 35,200 frames.",
         [str(PRETRAINED.relative_to(ROOT)), str(RETRAINED.relative_to(ROOT))], "Main README")

    historical = evidence["official_log_archive"]
    historic = next(item for item in historical if item.get("frames") == 25472 and "202431124637" in item["path"])
    bar_plot(["Paper", "Author historical", "Official pretrained", "Our reproduction"],
             [22.636, historic["mpjpe_mm"], evidence["pretrained"]["official_pooled"]["mpjpe_mm"], evidence["retrained"]["official_pooled"]["mpjpe_mm"]],
             "Overall MPJPE Comparison", "MPJPE (mm)", "mpjpe_comparison_overall.png", records,
             ["Paper Figure 12(c) value", "audit_evidence.json"], "Main README", rotation=12)
    bar_plot(["Paper", "Author historical", "Official pretrained", "Our reproduction"],
             [90.41, historic["pck_4cm_percent"], evidence["pretrained"]["official_pooled"]["pck_4cm_percent"], evidence["retrained"]["official_pooled"]["pck_4cm_percent"]],
             "Overall PCK@4cm Comparison", "PCK@4cm (%)", "pck4cm_comparison_overall.png", records,
             ["Paper Figure 12(c) value", "audit_evidence.json"], "Main README", rotation=12)
    bar_plot(["Paper testing\nsamples", "Author historical\nframes", "Official current\ninference frames"], [34520, 25472, 35200],
             "Evaluated Frame Count by Protocol", "Frames", "evaluated_frame_count_comparison.png", records,
             ["Paper Section 5.1 value", "audit_evidence.json"], "Main README", rotation=0)

    shared = overlap["common_exact_label_frame_metrics_strictly_comparable"]
    bar_plot(["Author historical", "Official pretrained", "Our reproduction"],
             [shared["b_historical"]["mpjpe_mm"], shared["a_pretrained"]["mpjpe_mm"], shared["a_retrained"]["mpjpe_mm"]],
             "Shared 24,960-Frame Subset: MPJPE", "MPJPE (mm)", "shared_subset_performance.png", records,
             ["testset_overlap_analysis.json, exact-label matched shared subset"], "Main README", rotation=12)
    bar_plot(["Author historical", "Official pretrained", "Our reproduction"],
             [shared["b_historical"]["pck_4cm_percent"], shared["a_pretrained"]["pck_4cm_percent"], shared["a_retrained"]["pck_4cm_percent"]],
             "Shared 24,960-Frame Subset: PCK@4cm", "PCK@4cm (%)", "shared_subset_pck4cm.png", records,
             ["testset_overlap_analysis.json, exact-label matched shared subset"], "Analysis section", rotation=12)

    accounting = overlap["frame_accounting"]
    bar_plot(["Shared", "A-only\n(current inference)", "B-only\n(historical logs)"],
             [accounting["exactly_matched_common_frames"], accounting["a_only_file_frames"], accounting["b_only_file_frames"]],
             "Testset Composition: Current Inference (A) vs Historical Logs (B)", "Frames", "testset_overlap_summary.png", records,
             ["testset_overlap_analysis.json"], "Main README", rotation=0)

    category_order = ["standard_x_x", "covering", "light_condition", "distance", "hand_temperature", "ambient_temperature", "ambient_interference"]
    category_names = {"standard_x_x": "standard_x_x", "covering": "covering", "light_condition": "light condition", "distance": "distance",
                      "hand_temperature": "hand temperature", "ambient_temperature": "ambient temperature", "ambient_interference": "ambient interference"}
    category_map = {row["value"]: row for row in categories if row["dimension"] == "category"}
    labels = [category_names[key] for key in category_order]
    a_frames = [float(category_map[key]["a_frames"] or 0) for key in category_order]
    b_frames = [float(category_map[key]["b_frames"] or 0) for key in category_order]
    x = np.arange(len(labels)); width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.1))
    ax.bar(x - width / 2, a_frames, width, label="A current inference")
    ax.bar(x + width / 2, b_frames, width, label="B historical logs")
    ax.set(xticks=x, xticklabels=labels, ylabel="Frames", title="Testset Frame Counts by Filename Category")
    ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=0.3); ax.legend()
    save(fig, "testset_category_frame_counts.png", records, "A/B frame counts by mechanically audited filename category.",
         ["testset_category_summary.csv"], "Analysis section")

    official_mpjpe = [float(category_map[key]["a_mpjpe_mm"]) for key in category_order]
    ours_mpjpe = [float(category_map[key]["a_retrained_mpjpe_mm"]) for key in category_order]
    fig, ax = plt.subplots(figsize=(10, 5.1))
    ax.bar(x - width / 2, official_mpjpe, width, label="Official pretrained")
    ax.bar(x + width / 2, ours_mpjpe, width, label="Our reproduction")
    ax.set(xticks=x, xticklabels=labels, ylabel="MPJPE (mm)", title="MPJPE by Filename Category")
    ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=0.3); ax.legend()
    save(fig, "category_mpjpe_official_vs_ours.png", records, "Independent-inference MPJPE by audited filename category.",
         ["testset_category_summary.csv"], "Analysis section")
    official_pck = [float(category_map[key]["a_pck_4cm_percent"]) for key in category_order]
    ours_pck_category = [float(category_map[key]["a_retrained_pck_4cm_percent"]) for key in category_order]
    fig, ax = plt.subplots(figsize=(10, 5.1))
    ax.bar(x - width / 2, official_pck, width, label="Official pretrained")
    ax.bar(x + width / 2, ours_pck_category, width, label="Our reproduction")
    ax.set(xticks=x, xticklabels=labels, ylabel="PCK@4cm (%)", title="PCK@4cm by Filename Category")
    ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=0.3); ax.legend()
    save(fig, "category_pck4cm_official_vs_ours.png", records, "Independent-inference PCK@4cm by audited filename category.",
         ["testset_category_summary.csv"], "Analysis section")

    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    joint = np.arange(21)
    ax.plot(joint, pre_error.mean(axis=0) * 1000, marker="o", label="Official pretrained")
    ax.plot(joint, ours_error.mean(axis=0) * 1000, marker="o", label="Our reproduction")
    ax.set(xlabel="Joint index", ylabel="Mean joint error (mm)", title="Mean Error Across 21 Hand Joints", xticks=joint)
    ax.grid(True, alpha=0.3); ax.legend()
    save(fig, "joint_error_comparison.png", records, "All-35,200-frame mean Euclidean error for every joint.",
         [str(PRETRAINED.relative_to(ROOT)), str(RETRAINED.relative_to(ROOT))], "Analysis section")

    users = [row for row in categories if row["dimension"] == "participant" and float(row["a_frames"] or 0) > 0]
    users.sort(key=lambda row: (int(row["value"][1:]) if row["value"].startswith("U") and row["value"][1:].isdigit() else 999, row["value"]))
    labels_users = [row["value"] for row in users]
    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    x_user = np.arange(len(users))
    ax.bar(x_user - width / 2, [float(row["a_mpjpe_mm"]) for row in users], width, label="Official pretrained")
    ax.bar(x_user + width / 2, [float(row["a_retrained_mpjpe_mm"]) for row in users], width, label="Our reproduction")
    ax.set(xticks=x_user, xticklabels=labels_users, ylabel="MPJPE (mm)", title="MPJPE by Filename User Token")
    ax.grid(axis="y", alpha=0.3); ax.legend()
    save(fig, "per_user_mpjpe.png", records, "A-only/current-inference per-user-token aggregation; conditions differ by user.",
         ["testset_category_summary.csv"], "Analysis section")

    for error, filename, title, label in [
        (pre_error, "framewise_mpjpe_histogram_official.png", "Framewise MPJPE Distribution: Official Pretrained", "Official pretrained"),
        (ours_error, "framewise_mpjpe_histogram_ours.png", "Framewise MPJPE Distribution: Our Reproduction", "Our reproduction"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        ax.hist(error.mean(axis=1) * 1000, bins=50, label=label)
        ax.set(xlabel="Framewise MPJPE (mm)", ylabel="Frames", title=title)
        ax.grid(axis="y", alpha=0.3); ax.legend()
        save(fig, filename, records, "Distribution of per-frame mean 21-joint errors over the independent 35,200-frame set.",
             [str(PRETRAINED.relative_to(ROOT)) if "Official" in label else str(RETRAINED.relative_to(ROOT))], "Analysis section")

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.boxplot([per_file_mpjpe(pre), per_file_mpjpe(ours)], labels=["Official pretrained", "Our reproduction"], showfliers=True)
    ax.set(ylabel="Per-file MPJPE (mm)", title="Per-file MPJPE Distribution")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "per_file_mpjpe_boxplot.png", records, "One MPJPE value per result_dict file (207 files per model).",
         [str(PRETRAINED.relative_to(ROOT)), str(RETRAINED.relative_to(ROOT))], "Analysis section")

    manifest = {"purpose": "Read-only figures generated from local TAPOR reproduction/audit artifacts.", "figures": records,
                "skipped_figures": [], "notes": ["Main comparison figures use independent Testset inference, not the training-internal random split.",
                           "Paper values are reported reference values, not recomputed from local predictions.",
                           "standard_x_x is a mechanical filename category, not a documented official definition of normal conditions."]}
    manifest_path = args.manifest or ROOT / "figure_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

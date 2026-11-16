#!/usr/bin/env python3
"""Generate a Markdown summary only from files that actually exist on the remote host."""
import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


METRICS = ("MPJPE", "PCK@5mm", "PCK@1cm", "PCK@2cm", "PCK@3cm", "PCK@4cm", "PCK@5cm", "PCK@6cm", "PCK@7cm", "PCK@8cm", "PCK@9cm")


def git_revision(root):
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unavailable"


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": f"{type(exc).__name__}: {exc}"}


def parse_metrics(text):
    found = {}
    for name in METRICS:
        values = re.findall(rf"{re.escape(name)}\s+([-+0-9.eE]+)", text)
        if values:
            found[name] = values[-1]
    for name in ("Average MSE Loss(testset)", "Average MAE Loss(testset)", "Mean root drift error"):
        values = re.findall(rf"{re.escape(name)}\s*[: ]\s*([-+0-9.eE]+)", text)
        if values:
            found[name] = values[-1]
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tapor-root", type=Path, default=Path("/workspace/TAPOR"))
    parser.add_argument("--runs-root", type=Path, default=Path("/root/sj-tmp/tapor_runs"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.runs_root / "EXPERIMENT_SUMMARY.md"
    manifests = sorted(args.runs_root.rglob("manifest.json")) if args.runs_root.exists() else []
    lines = ["# TAPOR Reproduction Experiment Summary", "", f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}", "", "## Provenance", "", f"- TAPOR source: `{args.tapor_root}`", f"- TAPOR Git revision: `{git_revision(args.tapor_root)}`", f"- Runs root: `{args.runs_root}`", "- This document reports only artifacts found on disk. Missing metrics mean they were not found; they are not inferred.", ""]
    reports = sorted(args.runs_root.rglob("dataset_inspection.json")) if args.runs_root.exists() else []
    lines += ["## Dataset Validation", ""]
    if not reports:
        lines.append("No dataset inspection report found.")
    for report_path in reports:
        report = read_json(report_path)
        lines += [f"### `{report_path}`", "", f"- Discovered pickle files: {report.get('discovered_pickle_count', 'unavailable')}", f"- Missing official train files: {len(report.get('official_train_missing', [])) if isinstance(report.get('official_train_missing'), list) else 'unavailable'}", f"- Missing official test files: {len(report.get('official_test_missing', [])) if isinstance(report.get('official_test_missing'), list) else 'unavailable'}", f"- Invalid/unreadable inspected files: {len(report.get('invalid_or_unreadable', [])) if isinstance(report.get('invalid_or_unreadable'), list) else 'unavailable'}", ""]
    lines += ["## Runs", ""]
    if not manifests:
        lines.append("No run manifests found.")
    for manifest_path in manifests:
        manifest = read_json(manifest_path)
        run_dir = manifest_path.parent
        lines += [f"### `{run_dir.name}`", ""]
        for key in ("kind", "command", "started_utc", "ended_utc", "elapsed_seconds", "exit_status"):
            lines.append(f"- {key}: `{manifest.get(key, 'unavailable')}`")
        console = run_dir / "console.log"
        metrics = parse_metrics(console.read_text(encoding="utf-8", errors="replace")) if console.exists() else {}
        if metrics:
            lines.append("- Last metric values parsed from real console output (upstream prints per test file; these are not an aggregate):")
            lines += [f"  - {key}: `{value}`" for key, value in metrics.items()]
        else:
            lines.append("- Metrics: no parsable metrics found in this run's console log.")
        lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

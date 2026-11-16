#!/usr/bin/env bash
# Shared safety helpers. This file is sourced by the runnable shell scripts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TAPOR_ROOT="${TAPOR_ROOT:-/workspace/TAPOR}"
DATA_ROOT="${DATA_ROOT:-/root/sj-tmp/tapor_data}"
RUNS_ROOT="${RUNS_ROOT:-/root/sj-tmp/tapor_runs}"

die() { echo "ERROR: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "Missing file: $1"; }
require_dir() { [[ -d "$1" ]] || die "Missing directory: $1"; }
timestamp() { date -u +%Y%m%dT%H%M%SZ; }
new_run_dir() { local kind="$1"; local path="$RUNS_ROOT/${kind}_$(timestamp)_$$"; mkdir -p "$path"; printf '%s\n' "$path"; }

ensure_link() {
  local target="$1" link="$2"
  require_dir "$target"
  if [[ -L "$link" ]]; then
    [[ "$(readlink -f "$link")" == "$(readlink -f "$target")" ]] || die "Existing link points elsewhere: $link -> $(readlink -f "$link")"
  elif [[ -e "$link" ]]; then
    die "Refusing to replace existing non-symlink path: $link"
  else
    ln -s "$target" "$link"
    echo "Created link: $link -> $target"
  fi
}

prepare_pose_data_links() {
  require_dir "$TAPOR_ROOT"
  require_dir "$DATA_ROOT/Trainset"
  require_dir "$DATA_ROOT/Testset"
  ensure_link "$DATA_ROOT/Trainset" "$TAPOR_ROOT/Trainset"
  ensure_link "$DATA_ROOT/Testset" "$TAPOR_ROOT/Testset"
}

prepare_upstream_artifact_links() {
  require_dir "$TAPOR_ROOT"
  local artifact_root="$RUNS_ROOT/upstream_outputs"
  mkdir -p "$artifact_root"
  for name in Runs LogTrainTest TestDataset_logs; do
    mkdir -p "$artifact_root/$name"
    ensure_link "$artifact_root/$name" "$TAPOR_ROOT/$name"
  done
}

check_space() {
  local available_kb
  available_kb="$(df -Pk "$RUNS_ROOT" | awk 'NR==2 {print $4}')"
  [[ -n "$available_kb" && "$available_kb" -ge 10485760 ]] || die "Less than 10 GiB available at $RUNS_ROOT filesystem"
}

write_manifest() {
  local path="$1" kind="$2" command="$3" start="$4" end="$5" status="$6"
  python - "$path" "$kind" "$command" "$start" "$end" "$status" "$TAPOR_ROOT" "$DATA_ROOT" "$RUNS_ROOT" <<'PY'
import json, sys
path, kind, command, start, end, status, tapor, data, runs = sys.argv[1:]
json.dump({"kind": kind, "command": command, "started_utc": start, "ended_utc": end,
           "elapsed_seconds": int(end) - int(start), "exit_status": int(status),
           "tapor_root": tapor, "data_root": data, "runs_root": runs}, open(path, "w"), indent=2)
PY
}

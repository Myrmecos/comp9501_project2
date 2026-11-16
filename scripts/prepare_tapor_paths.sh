#!/usr/bin/env bash
# Set up only safe symlinks required by upstream hard-coded paths.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

prepare_upstream_artifact_links
if [[ -d "$DATA_ROOT/Trainset" && -d "$DATA_ROOT/Testset" ]]; then
  prepare_pose_data_links
else
  echo "Dataset split directories are not both available; skipped Trainset/Testset links."
fi
echo "TAPOR path preparation complete."

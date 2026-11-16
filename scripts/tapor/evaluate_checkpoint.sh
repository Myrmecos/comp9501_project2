#!/usr/bin/env bash
# Evaluate a local trained TAPOR checkpoint with the unchanged upstream test script.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

[[ $# -eq 1 ]] || die "Usage: $0 <checkpoint filename in $TAPOR_ROOT/weights, or absolute path>"
input="$1"
if [[ "$input" = /* ]]; then
  require_file "$input"
  [[ "$(dirname "$input")" == "$TAPOR_ROOT/weights" ]] || die "Official inference only accepts checkpoints under $TAPOR_ROOT/weights"
  CHECKPOINT_NAME="$(basename "$input")"
else
  CHECKPOINT_NAME="$input"
  require_file "$TAPOR_ROOT/weights/$CHECKPOINT_NAME"
fi
exec "$SCRIPT_DIR/run_pretrained_inference.sh" "$CHECKPOINT_NAME"

#!/usr/bin/env bash
# Applies only the documented GPU UUID parsing fix required by official traintest.py.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TAPOR_ROOT="${TAPOR_ROOT:-/workspace/TAPOR}"
PATCH_FILE="$PROJECT_ROOT/patches/tapor-gpu-selection.patch"

[[ -f "$TAPOR_ROOT/traintest.py" ]] || { echo "TAPOR source not found: $TAPOR_ROOT" >&2; exit 1; }
[[ -f "$PATCH_FILE" ]] || { echo "Patch not found: $PATCH_FILE" >&2; exit 1; }

if git -C "$TAPOR_ROOT" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
  echo "Compatibility patch is already applied."
elif git -C "$TAPOR_ROOT" apply --check "$PATCH_FILE"; then
  git -C "$TAPOR_ROOT" apply "$PATCH_FILE"
  echo "Applied TAPOR GPU UUID parsing compatibility patch."
else
  echo "Patch does not apply cleanly. Inspect $TAPOR_ROOT/traintest.py before proceeding." >&2
  exit 1
fi

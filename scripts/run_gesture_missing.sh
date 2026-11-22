#!/usr/bin/env bash
# Train gesture heads for missing pose backbones (skips existing best.pt).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset PYTHONPATH || true
mkdir -p log

BACKBONES=("$@")
if [[ ${#BACKBONES[@]} -eq 0 ]]; then
  BACKBONES=(hand3d mediapipe nanotapor A B C D)
fi

EXP_ID="gesture_batch_missing"
if [[ -f "log/${EXP_ID}.pid" ]] && kill -0 "$(cat "log/${EXP_ID}.pid")" 2>/dev/null; then
  echo "SKIP gesture batch already running pid=$(cat log/${EXP_ID}.pid)"
  exit 0
fi

setsid nohup env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" -u scripts/train_gesture.py \
  --config config/gesture.yml \
  --split-json data/tapor/gesture_split.json \
  --device cuda:0 \
  --backbones "${BACKBONES[@]}" \
  --out-md log/GESTURE_RESULTS.md \
  >"log/${EXP_ID}.log" 2>&1 &
echo $! >"log/${EXP_ID}.pid"
echo "Started gesture batch pid=$(cat log/${EXP_ID}.pid) backbones=${BACKBONES[*]}"

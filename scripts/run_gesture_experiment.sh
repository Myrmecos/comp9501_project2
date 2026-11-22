#!/usr/bin/env bash
# Gesture recognition on frozen pose backbones. One process at a time on cuda:0.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset PYTHONPATH || true

mkdir -p log log/tb weights

echo "=== GPU before gesture experiment ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee log/gpu_before_gesture.txt

"$PYTHON" -u scripts/make_gesture_split.py

# Sequential: pose backbones + GT-joints oracle
nohup env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  "$PYTHON" -u scripts/train_gesture.py \
  --config config/gesture.yml \
  --split-json data/tapor/gesture_split.json \
  --device cuda:0 \
  --out-md log/GESTURE_RESULTS.md \
  >log/gesture_experiment.log 2>&1 &
echo $! >log/gesture_experiment.pid
echo "Launched gesture experiment pid=$(cat log/gesture_experiment.pid)"
echo "Monitor: tail -f log/gesture_experiment.log"
echo "Results: log/GESTURE_RESULTS.md"

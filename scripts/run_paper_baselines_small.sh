#!/usr/bin/env bash
# Train MediaPipe3d + NanoTapor on the same small split (parallel). Never kills foreign PIDs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH || true

mkdir -p log log/tb weights

echo "=== GPU before ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee log/gpu_before_paper_baselines.txt

run_one() {
  local model="$1"
  local exp="$2"
  local logfile="log/${exp}.log"
  echo "Starting $exp ($model) -> $logfile"
  nohup env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u main.py \
    --model "$model" \
    --exp-id "$exp" \
    --device cuda:0 \
    --max-minutes 55 \
    >"$logfile" 2>&1 &
  echo $! >"log/${exp}.pid"
  echo "  pid=$(cat "log/${exp}.pid")"
}

run_one mediapipe mediapipe_small
run_one nanotapor nanotapor_small

echo "All launched. Waiting..."
fail=0
for exp in mediapipe_small nanotapor_small; do
  if ! wait "$(cat "log/${exp}.pid")"; then
    echo "Process for $exp exited non-zero"
    fail=1
  fi
done

echo "=== GPU after ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv | tee log/gpu_after_paper_baselines.txt

env -u PYTHONPATH "$PYTHON" scripts/collect_paper_baselines.py || true
exit "$fail"

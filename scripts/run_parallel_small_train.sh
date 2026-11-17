#!/usr/bin/env bash
# Launch five parallel trains (mask-fixed + rebalanced). Never kills foreign PIDs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="/home/liuxuan/anaconda3/envs/py3.10/bin/python"
export CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH || true

mkdir -p log log/tb weights

echo "=== GPU before ==="
nvidia-smi | tee log/gpu_before_launch.txt
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee -a log/gpu_before_launch.txt

if [[ ! -f data/tapor/small_split.json ]]; then
  echo "Building small split..."
  env -u PYTHONPATH "$PYTHON" scripts/make_small_split.py
fi

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

run_one vanilla vanilla_maskfix_small
run_one A hybridA_maskfix_small
run_one B hybridB_maskfix_small
run_one C hybridC_maskfix_small
run_one D hybridD_maskfix_small

echo "All launched. Waiting..."
PIDS=()
for exp in vanilla_maskfix_small hybridA_maskfix_small hybridB_maskfix_small hybridC_maskfix_small hybridD_maskfix_small; do
  PIDS+=("$(cat "log/${exp}.pid")")
done

fail=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "Process $pid exited non-zero"
    fail=1
  fi
done

echo "=== GPU after ==="
nvidia-smi | tee log/gpu_after.txt

env -u PYTHONPATH "$PYTHON" scripts/collect_small_train_results.py || true
exit "$fail"

#!/usr/bin/env bash
# Launch 8 parallel full-dataset trains. Never kills foreign GPU PIDs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH || true

mkdir -p log log/tb weights

echo "=== GPU before ==="
nvidia-smi | tee log/gpu_before_full_train.txt
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee -a log/gpu_before_full_train.txt

if [[ ! -f data/tapor/full_split.json ]]; then
  echo "Building full_split.json ..."
  env -u PYTHONPATH "$PYTHON" scripts/make_full_split.py
fi

run_one() {
  local model="$1"
  local exp="$2"
  local logfile="log/${exp}.log"
  echo "Starting $exp ($model) -> $logfile"
  nohup env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u main.py \
    --model "$model" \
    --exp-id "$exp" \
    --config config/full.yml \
    --split-json data/tapor/full_split.json \
    --device cuda:0 \
    >"$logfile" 2>&1 &
  echo $! >"log/${exp}.pid"
  echo "  pid=$(cat "log/${exp}.pid")"
}

# Paper baselines + full TAPOR + full-power hybrids A–D
run_one hand3d hand3d_full
run_one mediapipe mediapipe_full
run_one nanotapor nanotapor_full
run_one vanilla vanilla_full
run_one A hybridA_full
run_one B hybridB_full
run_one C hybridC_full
run_one D hybridD_full

echo "All 8 launched. Waiting..."
EXPS=(hand3d_full mediapipe_full nanotapor_full vanilla_full hybridA_full hybridB_full hybridC_full hybridD_full)
fail=0
for exp in "${EXPS[@]}"; do
  if ! wait "$(cat "log/${exp}.pid")"; then
    echo "Process for $exp exited non-zero"
    fail=1
  fi
done

echo "=== GPU after ==="
nvidia-smi | tee log/gpu_after_full_train.txt

env -u PYTHONPATH "$PYTHON" scripts/collect_full_train_results.py || true
exit "$fail"

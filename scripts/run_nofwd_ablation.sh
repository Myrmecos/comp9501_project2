#!/usr/bin/env bash
# Ablation: hybrids A–D without ForwardThermal (no thermal recon loss).
# Same full_split / capacity as hybrid*_full; config/full_nofwd.yml sets use_forward: false.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH || true

mkdir -p log log/tb weights

echo "=== GPU before nofwd ablation ==="
nvidia-smi | tee log/gpu_before_nofwd_ablation.txt
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee -a log/gpu_before_nofwd_ablation.txt

if [[ ! -f data/tapor/full_split.json ]]; then
  echo "Building full_split.json ..."
  env -u PYTHONPATH "$PYTHON" scripts/make_full_split.py
fi

run_one() {
  local model="$1"
  local exp="$2"
  local logfile="log/${exp}.log"
  echo "Starting $exp ($model) -> $logfile"
  # setsid: survive SIGHUP when launcher exits
  setsid nohup env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u main.py \
    --model "$model" \
    --exp-id "$exp" \
    --config config/full_nofwd.yml \
    --split-json data/tapor/full_split.json \
    --device cuda:0 \
    >"$logfile" 2>&1 &
  echo $! >"log/${exp}.pid"
  echo "  pid=$(cat "log/${exp}.pid")"
}

run_one A hybridA_nofwd_full
run_one B hybridB_nofwd_full
run_one C hybridC_nofwd_full
run_one D hybridD_nofwd_full

echo "All 4 nofwd ablation jobs launched."
for exp in hybridA_nofwd_full hybridB_nofwd_full hybridC_nofwd_full hybridD_nofwd_full; do
  echo "  $exp pid=$(cat "log/${exp}.pid") log=log/${exp}.log"
done

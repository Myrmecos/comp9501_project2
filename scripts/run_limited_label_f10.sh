#!/usr/bin/env bash
# Limited-label f10 (TAP3D pretrain→finetune + nofwd ablation).
# Launches SSL + Nofwd for A–D, then a watcher per model that starts finetune
# after that model's SSL process exits.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="/home/liuxuan/anaconda3/envs/py3.10/bin/python"
export CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH || true

FRAC=10
SPLIT_SSL="data/tapor/limited_label_f${FRAC}_ssl.json"
SPLIT_CPL="data/tapor/limited_label_f${FRAC}_coupled.json"
CFG_SSL="config/full_ssl_pretrain.yml"
CFG_NOFWD="config/full_nofwd.yml"

mkdir -p log log/tb weights

for f in "$SPLIT_SSL" "$SPLIT_CPL" "$CFG_SSL" "$CFG_NOFWD" scripts/_ll_ft_watcher.sh; do
  [[ -f "$f" ]] || { echo "Missing $f"; exit 1; }
done

echo "=== GPU before limited-label f${FRAC} ==="
nvidia-smi | tee "log/gpu_before_ll_f${FRAC}.txt"

run_one() {
  local model="$1"
  local exp="$2"
  local cfg="$3"
  local split="$4"
  local logfile="log/${exp}.log"
  echo "Starting $exp ($model) -> $logfile"
  setsid nohup env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u main.py \
    --model "$model" \
    --exp-id "$exp" \
    --config "$cfg" \
    --split-json "$split" \
    --device cuda:0 \
    >"$logfile" 2>&1 &
  echo $! >"log/${exp}.pid"
  echo "  pid=$(cat "log/${exp}.pid")"
}

MODELS=(A B C D)
for m in "${MODELS[@]}"; do
  run_one "$m" "hybrid${m}_ll${FRAC}_ssl" "$CFG_SSL" "$SPLIT_SSL"
  run_one "$m" "hybrid${m}_ll${FRAC}_nofwd" "$CFG_NOFWD" "$SPLIT_CPL"
done

for m in "${MODELS[@]}"; do
  wlog="log/hybrid${m}_ll${FRAC}_ft_watcher.log"
  echo "Starting FT watcher for $m -> $wlog"
  setsid nohup bash "$ROOT/scripts/_ll_ft_watcher.sh" "$m" "$FRAC" >"$wlog" 2>&1 &
  echo $! >"log/hybrid${m}_ll${FRAC}_ft_watcher.pid"
  echo "  watcher_pid=$(cat "log/hybrid${m}_ll${FRAC}_ft_watcher.pid")"
done

echo
echo "Launched limited-label f${FRAC}:"
for m in "${MODELS[@]}"; do
  echo "  SSL   hybrid${m}_ll${FRAC}_ssl   pid=$(cat "log/hybrid${m}_ll${FRAC}_ssl.pid")"
  echo "  Nofwd hybrid${m}_ll${FRAC}_nofwd pid=$(cat "log/hybrid${m}_ll${FRAC}_nofwd.pid")"
  echo "  FT-watch hybrid${m}_ll${FRAC}_ft_watcher pid=$(cat "log/hybrid${m}_ll${FRAC}_ft_watcher.pid")"
done

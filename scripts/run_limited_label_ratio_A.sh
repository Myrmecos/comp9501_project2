#!/usr/bin/env bash
# Early (A) limited-label ratio sweep: SSL + Nofwd for given fracs, FT watchers.
# Does not touch hybridA_ll10_* (already running).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH || true

# Default: missing ratios for the 5/10/20/40/60 sweep (skip f10)
FRACS=("${@:-05 20 40 60}")

CFG_SSL="config/full_ssl_pretrain.yml"
CFG_NOFWD="config/full_nofwd.yml"
mkdir -p log log/tb weights

echo "=== GPU before Early ratio launch (${FRACS[*]}) ==="
nvidia-smi | tee log/gpu_before_ll_ratio_A.txt

run_one() {
  local model="$1"
  local exp="$2"
  local cfg="$3"
  local split="$4"
  local logfile="log/${exp}.log"
  if [[ -f "log/${exp}.pid" ]] && kill -0 "$(cat "log/${exp}.pid")" 2>/dev/null; then
    echo "SKIP running $exp pid=$(cat "log/${exp}.pid")"
    return 0
  fi
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

for pct in "${FRACS[@]}"; do
  ssl_split="data/tapor/limited_label_f${pct}_ssl.json"
  cpl_split="data/tapor/limited_label_f${pct}_coupled.json"
  [[ -f "$ssl_split" && -f "$cpl_split" ]] || { echo "Missing splits for f${pct}"; exit 1; }
  run_one A "hybridA_ll${pct}_ssl" "$CFG_SSL" "$ssl_split"
  run_one A "hybridA_ll${pct}_nofwd" "$CFG_NOFWD" "$cpl_split"

  wlog="log/hybridA_ll${pct}_ft_watcher.log"
  if [[ -f "log/hybridA_ll${pct}_ft_watcher.pid" ]] && kill -0 "$(cat "log/hybridA_ll${pct}_ft_watcher.pid")" 2>/dev/null; then
    echo "SKIP watcher hybridA_ll${pct}_ft_watcher"
  else
    echo "Starting FT watcher for A f${pct} -> $wlog"
    setsid nohup bash "$ROOT/scripts/_ll_ft_watcher.sh" A "$pct" >"$wlog" 2>&1 &
    echo $! >"log/hybridA_ll${pct}_ft_watcher.pid"
  fi
done

echo "Done launching Early ratio jobs for: ${FRACS[*]}"

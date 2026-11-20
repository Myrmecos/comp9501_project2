#!/usr/bin/env bash
# Paired modified_A full vs nofwd screens (new exp-ids only; never clobber).
# Usage: bash scripts/run_modified_a_screens.sh [S0 S1 ...]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="/home/liuxuan/anaconda3/envs/py3.10/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset PYTHONPATH || true
mkdir -p log log/tb weights

SCREENS=("${@:-S0}")
SPLIT="data/tapor/full_split.json"

run_one() {
  local exp="$1" cfg="$2"
  if [[ -f "log/${exp}.pid" ]] && kill -0 "$(cat "log/${exp}.pid")" 2>/dev/null; then
    echo "SKIP running $exp"
    return 0
  fi
  # Only skip finished runs (summary.json). Mid-train best.pt alone is not enough.
  if [[ -f "weights/${exp}/summary.json" ]]; then
    echo "SKIP finished $exp (summary.json)"
    return 0
  fi
  if [[ -f "weights/${exp}/best.pt" ]]; then
    echo "WARN $exp has best.pt but no summary — will not clobber; skip"
    return 0
  fi
  echo "Starting $exp cfg=$cfg"
  setsid nohup env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" -u main.py \
    --model modified_A \
    --exp-id "$exp" \
    --config "$cfg" \
    --split-json "$SPLIT" \
    --device cuda:0 \
    --max-epochs 40 \
    >"log/${exp}.log" 2>&1 &
  echo $! >"log/${exp}.pid"
  echo "  pid=$(cat "log/${exp}.pid")"
}

for s in "${SCREENS[@]}"; do
  sid="${s#S}"
  sid="${sid#s}"
  cfg="config/modified_a_scr${sid}.yml"
  cfg_n="config/modified_a_scr${sid}_nofwd.yml"
  [[ -f "$cfg" && -f "$cfg_n" ]] || { echo "Missing $cfg"; exit 1; }
  run_one "modifiedA_scr${sid}_full" "$cfg"
  run_one "modifiedA_scr${sid}_nofwd" "$cfg_n"
done

echo "Launched screens: ${SCREENS[*]}"

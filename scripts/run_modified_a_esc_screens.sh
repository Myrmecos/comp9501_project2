#!/usr/bin/env bash
# Escalated modified_A screens (mask + inv skip). New exp-ids only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="/home/liuxuan/anaconda3/envs/py3.10/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset PYTHONPATH || true
mkdir -p log weights

SCREENS=("${@:-S0}")
SPLIT="data/tapor/full_split.json"

run_one() {
  local exp="$1" cfg="$2"
  if [[ -f "log/${exp}.pid" ]] && kill -0 "$(cat "log/${exp}.pid")" 2>/dev/null; then
    echo "SKIP running $exp"; return 0
  fi
  if [[ -f "weights/${exp}/summary.json" ]]; then
    echo "SKIP finished $exp"; return 0
  fi
  if [[ -f "weights/${exp}/best.pt" ]]; then
    echo "WARN $exp has best.pt but no summary — skip (no clobber)"; return 0
  fi
  echo "Starting $exp cfg=$cfg"
  setsid nohup env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" -u main.py \
    --model modified_A --exp-id "$exp" --config "$cfg" --split-json "$SPLIT" \
    --device cuda:0 --max-epochs 40 \
    >"log/${exp}.log" 2>&1 &
  echo $! >"log/${exp}.pid"
  echo "  pid=$(cat "log/${exp}.pid")"
}

for s in "${SCREENS[@]}"; do
  sid="${s#S}"; sid="${sid#s}"
  cfg="config/modified_a_esc_scr${sid}.yml"
  cfg_n="config/modified_a_esc_scr${sid}_nofwd.yml"
  [[ -f "$cfg" && -f "$cfg_n" ]] || { echo "Missing $cfg"; exit 1; }
  run_one "modifiedA_esc_scr${sid}_full" "$cfg"
  run_one "modifiedA_esc_scr${sid}_nofwd" "$cfg_n"
done
echo "Launched ESC screens: ${SCREENS[*]}"

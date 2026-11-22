#!/usr/bin/env bash
# P4 Forward vs nofwd across labeled fractions 5/10/20/60%.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SW="$ROOT/physics_correct_ws"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset PYTHONPATH || true
mkdir -p "$SW/log" "$SW/weights"
ART="$SW"

active_exps() {
  # Avoid pipefail exit when grep finds no matches (idle GPU).
  local n
  n=$(ps aux | grep -E 'main.py .*--exp-id' | grep -v grep | grep -oE -- '--exp-id [^ ]+' | awk '{print $2}' | sort -u | wc -l || true)
  echo "${n:-0}"
}

wait_slots() {
  local max="${1:-2}"
  while true; do
    local n; n=$(active_exps)
    echo "[frac] active_exps=$n (max=$max)"
    [[ "$n" -le "$max" ]] && break
    sleep 90
  done
}

run_one() {
  local exp="$1" cfg="$2" split="$3"
  if [[ -f "$SW/weights/${exp}/summary.json" ]]; then
    echo "SKIP finished $exp"; return 0
  fi
  if [[ -f "$SW/log/${exp}.pid" ]] && kill -0 "$(cat "$SW/log/${exp}.pid")" 2>/dev/null; then
    echo "SKIP running $exp"; return 0
  fi
  echo "[frac] start $exp split=$(basename "$split")"
  setsid nohup env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" -u main.py \
    --model modified_A --exp-id "$exp" --config "$cfg" --split-json "$split" \
    --device cuda:0 --artifact-root "$ART" \
    >"$SW/log/${exp}.log" 2>&1 &
  echo $! >"$SW/log/${exp}.pid"
  echo "  pid=$(cat "$SW/log/${exp}.pid")"
}

wait_exp() {
  local exp="$1" pid
  pid=$(cat "$SW/log/${exp}.pid" 2>/dev/null || true)
  while true; do
    [[ -f "$SW/weights/${exp}/summary.json" ]] && break
    if [[ -n "${pid:-}" ]] && ! kill -0 "$pid" 2>/dev/null; then break; fi
    sleep 60
  done
}

run_frac() {
  local frac="$1"
  local split="$ROOT/data/tapor/limited_label_f${frac}_coupled.json"
  local exp="P4_f${frac}"
  wait_slots 3
  run_one "$exp" "$SW/config/${exp}.yml" "$split"
  run_one "${exp}_nofwd" "$SW/config/${exp}_nofwd.yml" "$split"
  wait_exp "$exp"
  wait_exp "${exp}_nofwd"
  "$PYTHON" "$SW/scripts/collect_label_frac.py" || true
}

for frac in 05 10 20 60; do
  run_frac "$frac"
done
"$PYTHON" "$SW/scripts/collect_label_frac.py"
echo "[frac] sweep done"

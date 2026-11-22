#!/usr/bin/env bash
# Paired micro screens under physics_correct_ws/
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SW="$ROOT/physics_correct_ws"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset PYTHONPATH || true
mkdir -p "$SW/log" "$SW/weights"

SPLIT="$SW/data/micro_split.json"
ART="$SW"
IDS=("${@:-P0}")

model_from_cfg() {
  local cfg="$1"
  local m
  m=$(grep -E '^phypose_model:' "$cfg" | awk '{print $2}' | tr -d '"' | tr -d "'")
  echo "${m:-modified_A}"
}

run_one() {
  local exp="$1" cfg="$2"
  local model
  model=$(model_from_cfg "$cfg")
  if [[ -f "$SW/log/${exp}.pid" ]] && kill -0 "$(cat "$SW/log/${exp}.pid")" 2>/dev/null; then
    echo "SKIP running $exp"; return 0
  fi
  if [[ -f "$SW/weights/${exp}/summary.json" ]]; then
    echo "SKIP finished $exp"; return 0
  fi
  echo "Starting $exp model=$model cfg=$cfg"
  setsid nohup env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" -u main.py \
    --model "$model" --exp-id "$exp" --config "$cfg" --split-json "$SPLIT" \
    --device cuda:0 --artifact-root "$ART" \
    >"$SW/log/${exp}.log" 2>&1 &
  echo $! >"$SW/log/${exp}.pid"
  echo "  pid=$(cat "$SW/log/${exp}.pid")"
}

wait_exp() {
  local exp="$1"
  local pid
  pid=$(cat "$SW/log/${exp}.pid" 2>/dev/null || true)
  while true; do
    [[ -f "$SW/weights/${exp}/summary.json" ]] && break
    if [[ -n "${pid:-}" ]] && ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 20
  done
}

for id in "${IDS[@]}"; do
  cfg="$SW/config/${id}.yml"
  cfg_n="$SW/config/${id}_nofwd.yml"
  [[ -f "$cfg" && -f "$cfg_n" ]] || { echo "Missing $cfg"; exit 1; }
  run_one "$id" "$cfg"
  run_one "${id}_nofwd" "$cfg_n"
  wait_exp "$id"
  wait_exp "${id}_nofwd"
  "$PYTHON" "$SW/scripts/collect_micro.py" >/dev/null || true
  echo "=== after $id ==="
  cat "$SW/log/MICRO_SCREEN.md" || true
done
echo "physics_correct micro batch done: ${IDS[*]}"

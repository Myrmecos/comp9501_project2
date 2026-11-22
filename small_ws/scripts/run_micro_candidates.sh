#!/usr/bin/env bash
# Paired micro screens under small_ws/. Usage: bash small_ws/scripts/run_micro_candidates.sh C0 C1 ...
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SW="$ROOT/small_ws"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset PYTHONPATH || true
mkdir -p "$SW/log" "$SW/weights"

SPLIT="$SW/data/micro_split.json"
ART="$SW"
IDS=("${@:-C0}")

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
  if [[ -f "$SW/weights/${exp}/best.pt" ]]; then
    echo "WARN $exp best.pt without summary — skip"; return 0
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
    sleep 30
  done
}

run_c4() {
  # SSL then FT for full and nofwd
  run_one C4_ssl "$SW/config/C4_ssl.yml"
  run_one C4_ssl_nofwd "$SW/config/C4_ssl_nofwd.yml"
  wait_exp C4_ssl
  wait_exp C4_ssl_nofwd
  # FT from SSL best
  if [[ ! -f "$SW/weights/C4_ft/summary.json" ]]; then
    echo "Starting C4_ft from C4_ssl"
    setsid nohup env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" -u main.py \
      --model modified_A --exp-id C4_ft --config "$SW/config/C4_ft.yml" --split-json "$SPLIT" \
      --device cuda:0 --artifact-root "$ART" \
      --load-weights "$SW/weights/C4_ssl/best.pt" \
      >"$SW/log/C4_ft.log" 2>&1 &
    echo $! >"$SW/log/C4_ft.pid"
  fi
  if [[ ! -f "$SW/weights/C4_ft_nofwd/summary.json" ]]; then
    echo "Starting C4_ft_nofwd from C4_ssl_nofwd"
    setsid nohup env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" -u main.py \
      --model modified_A --exp-id C4_ft_nofwd --config "$SW/config/C4_ft_nofwd.yml" --split-json "$SPLIT" \
      --device cuda:0 --artifact-root "$ART" \
      --load-weights "$SW/weights/C4_ssl_nofwd/best.pt" \
      >"$SW/log/C4_ft_nofwd.log" 2>&1 &
    echo $! >"$SW/log/C4_ft_nofwd.pid"
  fi
  wait_exp C4_ft
  wait_exp C4_ft_nofwd
}

for id in "${IDS[@]}"; do
  if [[ "$id" == "C4" ]]; then
    run_c4
  else
    cfg="$SW/config/${id}.yml"
    cfg_n="$SW/config/${id}_nofwd.yml"
    [[ -f "$cfg" && -f "$cfg_n" ]] || { echo "Missing $cfg"; exit 1; }
    run_one "$id" "$cfg"
    run_one "${id}_nofwd" "$cfg_n"
    wait_exp "$id"
    wait_exp "${id}_nofwd"
  fi
  "$PYTHON" "$SW/scripts/collect_micro.py" >/dev/null || true
  echo "=== after $id ==="
  cat "$SW/log/MICRO_SCREEN.md" || true
done
echo "micro batch done: ${IDS[*]}"

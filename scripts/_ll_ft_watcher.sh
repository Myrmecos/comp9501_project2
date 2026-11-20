#!/usr/bin/env bash
# Wait for SSL job to exit, then start Stage-2 finetune with --load-weights.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="/home/liuxuan/anaconda3/envs/py3.10/bin/python"

MODEL="${1:?model}"
FRAC="${2:?frac}"
SSL_EXP="hybrid${MODEL}_ll${FRAC}_ssl"
FT_EXP="hybrid${MODEL}_ll${FRAC}_ft"
SSL_PID_FILE="log/${SSL_EXP}.pid"
SSL_BEST="weights/${SSL_EXP}/best.pt"
SPLIT_CPL="data/tapor/limited_label_f${FRAC}_coupled.json"
CFG_FT="config/full.yml"

if [[ ! -f "$SSL_PID_FILE" ]]; then
  echo "[watcher] missing $SSL_PID_FILE" >&2
  exit 1
fi
SSL_PID="$(cat "$SSL_PID_FILE")"
echo "[watcher] waiting for $SSL_EXP pid=$SSL_PID"

while kill -0 "$SSL_PID" 2>/dev/null; do
  sleep 60
done

# Give filesystem a moment to flush best.pt / summary
sleep 5
if [[ ! -f "$SSL_BEST" ]]; then
  echo "[watcher] ERROR: $SSL_BEST missing after SSL exit" | tee -a "log/${FT_EXP}.log"
  exit 1
fi

echo "[watcher] starting $FT_EXP from $SSL_BEST"
setsid nohup env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u main.py \
  --model "$MODEL" \
  --exp-id "$FT_EXP" \
  --config "$CFG_FT" \
  --split-json "$SPLIT_CPL" \
  --device cuda:0 \
  --load-weights "$SSL_BEST" \
  >"log/${FT_EXP}.log" 2>&1 &
echo $! >"log/${FT_EXP}.pid"
echo "[watcher] $FT_EXP pid=$(cat "log/${FT_EXP}.pid")"
wait "$(cat "log/${FT_EXP}.pid")"
echo "[watcher] $FT_EXP finished"

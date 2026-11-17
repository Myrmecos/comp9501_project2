#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="/home/liuxuan/anaconda3/envs/py3.10/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
unset PYTHONPATH || true

EXP=hand3d_noleak
LOG=log/${EXP}.log
echo "===== HAND3D NOLEAK TRAIN $(date -Is) =====" | tee -a "$LOG"
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONUNBUFFERED=1 \
  "$PYTHON" -u main.py \
  --model hand3d \
  --exp-id "$EXP" \
  --config config/full.yml \
  --split-json data/tapor/full_split_noleak.json \
  --device cuda:0 \
  --max-epochs 200 \
  >>"$LOG" 2>&1
rc=$?
echo "===== TRAIN DONE $(date -Is) exit=$rc =====" | tee -a "$LOG"
if [ "$rc" -ne 0 ]; then
  echo "train failed; skip eval" | tee -a "$LOG"
  exit "$rc"
fi

echo "===== OFFICIAL TEST EVAL $(date -Is) =====" | tee -a "$LOG"
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONUNBUFFERED=1 \
  "$PYTHON" -u scripts/eval_official_testset.py \
  --device cuda:0 \
  --batch-size 8 \
  --only hand3d_noleak \
  --merge \
  --out log/OFFICIAL_TEST_RESULTS.md \
  >>"$LOG" 2>&1
echo "===== PIPELINE DONE $(date -Is) =====" | tee -a "$LOG"

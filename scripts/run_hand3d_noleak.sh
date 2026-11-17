#!/usr/bin/env bash
# Train Hand3d from scratch on leak-audited file-disjoint split; then official Test eval.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="/home/liuxuan/anaconda3/envs/py3.10/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
unset PYTHONPATH || true

mkdir -p log log/tb weights

"$PYTHON" -u scripts/make_noleak_split.py \
  --out data/tapor/full_split_noleak.json \
  --audit-out log/DATA_LEAK_AUDIT.md

INNER="$ROOT/scripts/_hand3d_noleak_pipeline.sh"
cat >"$INNER" <<'INNER'
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
INNER
chmod +x "$INNER"

setsid nohup bash "$INNER" >>log/hand3d_noleak_runner.log 2>&1 < /dev/null &
echo $! | tee log/hand3d_noleak.pid
echo "Launched Hand3d no-leak pipeline pid=$(cat log/hand3d_noleak.pid)"
echo "Audit: log/DATA_LEAK_AUDIT.md"
echo "Monitor: tail -f log/hand3d_noleak.log"

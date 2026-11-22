#!/usr/bin/env bash
# Resume hybrid A–D sequentially (one GPU job at a time) to avoid OOM.
# Never kills foreign GPU PIDs. Appends to existing logs; TB appends new event files.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
unset PYTHONPATH || true

mkdir -p log log/tb weights

echo "=== GPU before sequential resume ===" | tee log/gpu_before_hybrid_resume.txt
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee -a log/gpu_before_hybrid_resume.txt
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv | tee -a log/gpu_before_hybrid_resume.txt

# B first (still improving @196); then A/C/D; then gesture
MODELS=(
  "B hybridB_full"
  "A hybridA_full"
  "C hybridC_full"
  "D hybridD_full"
)

run_one() {
  local model="$1"
  local exp="$2"
  local logfile="log/${exp}.log"
  echo "" | tee -a "$logfile"
  echo "===== RESUME $(date -Is) model=$model exp=$exp max_epochs=250 =====" | tee -a "$logfile"
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONUNBUFFERED=1 \
    "$PYTHON" -u main.py \
    --model "$model" \
    --exp-id "$exp" \
    --config config/full.yml \
    --split-json data/tapor/full_split.json \
    --device cuda:0 \
    --resume last \
    --max-epochs 250 \
    >>"$logfile" 2>&1
  local rc=$?
  echo "===== DONE $(date -Is) exp=$exp exit=$rc =====" | tee -a "$logfile"
  return $rc
}

PIPELINE="$ROOT/scripts/_resume_pipeline_inner.sh"
cat >"$PIPELINE" <<'INNER'
#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
unset PYTHONPATH || true

MODELS=(
  "B hybridB_full"
  "A hybridA_full"
  "C hybridC_full"
  "D hybridD_full"
)

run_one() {
  local model="$1"
  local exp="$2"
  local logfile="log/${exp}.log"
  echo "" | tee -a "$logfile"
  echo "===== RESUME $(date -Is) model=$model exp=$exp max_epochs=250 =====" | tee -a "$logfile"
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONUNBUFFERED=1 \
    "$PYTHON" -u main.py \
    --model "$model" \
    --exp-id "$exp" \
    --config config/full.yml \
    --split-json data/tapor/full_split.json \
    --device cuda:0 \
    --resume last \
    --max-epochs 250 \
    >>"$logfile" 2>&1
  local rc=$?
  echo "===== DONE $(date -Is) exp=$exp exit=$rc =====" | tee -a "$logfile"
  return $rc
}

echo "Sequential hybrid resume started at $(date -Is)"
for pair in "${MODELS[@]}"; do
  set -- $pair
  model="$1"
  exp="$2"
  echo "[runner] starting $exp ..."
  if run_one "$model" "$exp"; then
    echo "[runner] $exp finished OK"
  else
    echo "[runner] $exp failed (exit $?) — continuing"
  fi
  sleep 5
done
echo "All hybrid resumes finished at $(date -Is)"

echo "[runner] starting gesture experiment ..."
"$PYTHON" -u scripts/make_gesture_split.py >>log/gesture_experiment.log 2>&1 || true
if env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONUNBUFFERED=1 \
  "$PYTHON" -u scripts/train_gesture.py \
  --config config/gesture.yml \
  --split-json data/tapor/gesture_split.json \
  --device cuda:0 \
  --out-md log/GESTURE_RESULTS.md \
  >>log/gesture_experiment.log 2>&1
then
  echo "[runner] gesture finished OK"
else
  echo "[runner] gesture failed"
fi
echo "Pipeline complete at $(date -Is)"
INNER
chmod +x "$PIPELINE"

# setsid+nohup: survive launcher exit (previous resumes died from SIGHUP)
setsid nohup bash "$PIPELINE" >>log/hybrid_resume_runner.log 2>&1 < /dev/null &
echo $! | tee log/hybrid_resume_runner.pid
echo "Launched sequential resume+gesture pid=$(cat log/hybrid_resume_runner.pid)"
echo "Monitor: tail -f log/hybrid_resume_runner.log log/hybridB_full.log"
echo "TB: http://127.0.0.1:6006  (log/tb/)"

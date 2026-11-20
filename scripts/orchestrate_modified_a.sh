#!/usr/bin/env bash
# Run modified_A screens S0..S6 in order; stop early if gap>=0.08; then full-budget train.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="/home/liuxuan/anaconda3/envs/py3.10/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset PYTHONPATH || true
mkdir -p log weights config

GAP_TARGET=0.08
SOFT_GAP=0.05
SPLIT="data/tapor/full_split.json"

wait_pair() {
  local sid="$1"
  local fpid nfid
  fpid=$(cat "log/modifiedA_scr${sid}_full.pid" 2>/dev/null || true)
  nfid=$(cat "log/modifiedA_scr${sid}_nofwd.pid" 2>/dev/null || true)
  echo "[orch] waiting for S${sid} pair (pids ${fpid:-?} ${nfid:-?})"
  while true; do
    local alive=0
    if [[ -n "${fpid:-}" ]] && kill -0 "$fpid" 2>/dev/null; then alive=1; fi
    if [[ -n "${nfid:-}" ]] && kill -0 "$nfid" 2>/dev/null; then alive=1; fi
    # also done if both best.pt exist
    if [[ -f "weights/modifiedA_scr${sid}_full/best.pt" && -f "weights/modifiedA_scr${sid}_nofwd/best.pt" ]]; then
      if [[ "$alive" -eq 0 ]]; then
        echo "[orch] S${sid} both best.pt present and idle"
        break
      fi
    fi
    if [[ "$alive" -eq 0 ]]; then
      echo "[orch] S${sid} processes exited"
      break
    fi
    sleep 60
  done
  "$PYTHON" scripts/collect_modified_a_screens.py >/dev/null
}

best_gap_sid() {
  "$PYTHON" - <<'PY'
import json, math
from pathlib import Path
rows = json.loads(Path("log/modified_a_screens.json").read_text())
best = None
for r in rows:
    g = r.get("gap")
    if g is None or not math.isfinite(g):
        continue
    if best is None or g > best[0]:
        best = (g, r["sid"])
print(f"{best[1]} {best[0]}" if best else "-1 -999")
PY
}

promote_and_full() {
  local sid="$1"
  echo "[orch] promoting S${sid} -> modified_a_best.yml"
  cp -n "config/modified_a_scr${sid}.yml" "config/modified_a_best.yml" 2>/dev/null || cp "config/modified_a_scr${sid}.yml" "config/modified_a_best.yml"
  cp -n "config/modified_a_scr${sid}_nofwd.yml" "config/modified_a_best_nofwd.yml" 2>/dev/null || cp "config/modified_a_scr${sid}_nofwd.yml" "config/modified_a_best_nofwd.yml"
  # bump to full budget
  "$PYTHON" - <<'PY'
from pathlib import Path
import re
for name in ["config/modified_a_best.yml", "config/modified_a_best_nofwd.yml"]:
    p = Path(name)
    t = p.read_text()
    t = re.sub(r"(?m)^max_epochs:\s*\d+", "max_epochs: 200", t)
    t = re.sub(r"(?m)^early_stop_patience:\s*\d+", "early_stop_patience: 25", t)
    # keep phase1 from screen; ensure comment
    if "promoted from screen" not in t:
        t = "# promoted from screen for full-budget train\n" + t
    p.write_text(t)
print("wrote best configs")
PY

  run_full() {
    local exp="$1" cfg="$2"
    if [[ -f "weights/${exp}/best.pt" ]]; then
      echo "SKIP existing $exp"
      return 0
    fi
    if [[ -f "log/${exp}.pid" ]] && kill -0 "$(cat "log/${exp}.pid")" 2>/dev/null; then
      echo "SKIP running $exp"
      return 0
    fi
    setsid nohup env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" -u main.py \
      --model modified_A --exp-id "$exp" --config "$cfg" --split-json "$SPLIT" --device cuda:0 \
      >"log/${exp}.log" 2>&1 &
    echo $! >"log/${exp}.pid"
    echo "Started $exp pid=$(cat log/${exp}.pid)"
  }
  run_full modifiedA_full config/modified_a_best.yml
  run_full modifiedA_nofwd_full config/modified_a_best_nofwd.yml
}

# Ensure S0 launched
bash scripts/run_modified_a_screens.sh S0

BEST_SID=-1
BEST_GAP=-999
for sid in 0 1 2 3 4 5 6; do
  bash scripts/run_modified_a_screens.sh "S${sid}"
  wait_pair "$sid"
  read -r BEST_SID BEST_GAP <<<"$(best_gap_sid)"
  echo "[orch] after S${sid}: best_sid=${BEST_SID} best_gap=${BEST_GAP}"
  # stop early if current or best gap hits target
  CUR=$("$PYTHON" - <<PY
import json, math
from pathlib import Path
rows = json.loads(Path("log/modified_a_screens.json").read_text())
r = rows[${sid}]
g = r.get("gap")
print(g if g is not None and math.isfinite(g) else -999)
PY
)
  echo "[orch] S${sid} gap=${CUR}"
  # early stop if this screen or overall best >= target
  awk -v g="$BEST_GAP" -v t="$GAP_TARGET" 'BEGIN{exit !(g+0>=t+0)}' && {
    echo "[orch] early stop: best gap ${BEST_GAP} >= ${GAP_TARGET}"
    break
  }
done

read -r BEST_SID BEST_GAP <<<"$(best_gap_sid)"
echo "[orch] FINAL best S${BEST_SID} gap=${BEST_GAP}"
if awk -v g="$BEST_GAP" -v t="$SOFT_GAP" 'BEGIN{exit !(g+0>=t+0)}'; then
  promote_and_full "$BEST_SID"
else
  echo "[orch] gap < ${SOFT_GAP}; escalate: soft_depth + mask-like defaults still use best positive if any"
  if [[ "$BEST_SID" -ge 0 ]]; then
    promote_and_full "$BEST_SID"
  else
    promote_and_full 0
  fi
fi

# wait for full train pair
wait_full() {
  local exp="$1"
  local pid
  pid=$(cat "log/${exp}.pid" 2>/dev/null || true)
  echo "[orch] waiting $exp"
  while true; do
    if [[ -f "weights/${exp}/summary.json" ]]; then
      if [[ -z "${pid:-}" ]] || ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
    fi
    if [[ -n "${pid:-}" ]] && ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 120
  done
}
wait_full modifiedA_full
wait_full modifiedA_nofwd_full

# official test (preserve prior rows via --merge)
"$PYTHON" -u scripts/eval_official_testset.py \
  --only modifiedA_full modifiedA_nofwd_full --merge || true

"$PYTHON" scripts/collect_modified_a_screens.py
echo "[orch] DONE screens+full"

# Gesture heads for missing pose backbones (+ modifiedA when ready)
BACKBONES=(hand3d mediapipe nanotapor A B C D)
if [[ -f weights/modifiedA_full/best.pt ]]; then
  BACKBONES+=(modified_A)
fi
echo "[orch] launching gesture for: ${BACKBONES[*]}"
setsid nohup env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" -u scripts/train_gesture.py \
  --config config/gesture.yml \
  --split-json data/tapor/gesture_split.json \
  --device cuda:0 \
  --backbones "${BACKBONES[@]}" \
  --out-md log/GESTURE_RESULTS.md \
  >log/gesture_batch_missing.log 2>&1 &
echo $! > log/gesture_batch_missing.pid
gpid=$(cat log/gesture_batch_missing.pid)
echo "[orch] gesture pid=$gpid"
while kill -0 "$gpid" 2>/dev/null; do sleep 60; done

echo done > log/modified_a_orch.done
echo "[orch] ALL DONE"

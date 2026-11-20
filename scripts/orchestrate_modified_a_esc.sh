#!/usr/bin/env bash
# Escalate modified_A (mask+inv skip): screens S0,S1,S3 → full train → official test → gesture.
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
  fpid=$(cat "log/modifiedA_esc_scr${sid}_full.pid" 2>/dev/null || true)
  nfid=$(cat "log/modifiedA_esc_scr${sid}_nofwd.pid" 2>/dev/null || true)
  echo "[esc] waiting S${sid} (pids ${fpid:-?} ${nfid:-?})"
  while true; do
    local alive=0
    [[ -n "${fpid:-}" ]] && kill -0 "$fpid" 2>/dev/null && alive=1
    [[ -n "${nfid:-}" ]] && kill -0 "$nfid" 2>/dev/null && alive=1
    if [[ -f "weights/modifiedA_esc_scr${sid}_full/summary.json" && -f "weights/modifiedA_esc_scr${sid}_nofwd/summary.json" ]]; then
      [[ "$alive" -eq 0 ]] && break
    fi
    [[ "$alive" -eq 0 ]] && break
    sleep 60
  done
  "$PYTHON" scripts/collect_modified_a_screens.py >/dev/null || true
}

best_gap_sid() {
  "$PYTHON" - <<'PY'
import json, math
from pathlib import Path
rows = []
for sid in (0, 1, 3):
    def best(exp):
        p = Path(f"weights/{exp}/summary.json")
        if not p.exists():
            return None
        b = (json.loads(p.read_text()).get("best") or {})
        m = b.get("mpjpe_cm")
        return float(m) if m is not None else None
    f, n = best(f"modifiedA_esc_scr{sid}_full"), best(f"modifiedA_esc_scr{sid}_nofwd")
    gap = (n - f) if f is not None and n is not None else None
    rows.append((gap, sid, f, n))
best = None
for g, sid, f, n in rows:
    if g is None or not math.isfinite(g):
        continue
    if best is None or g > best[0]:
        best = (g, sid)
print(f"{best[1]} {best[0]}" if best else "-1 -999")
PY
}

append_esc_md() {
  "$PYTHON" - <<'PY'
import json, math
from pathlib import Path
lines = ["", "## Escalation (mask + inverse-feat skip)", "",
         "| screen | full MPJPE | nofwd MPJPE | gap | status |",
         "|--------|------------|-------------|-----|--------|"]
best_g, best_s = -1e9, None
for sid in (0, 1, 3):
    def best(exp):
        p = Path(f"weights/{exp}/summary.json")
        if not p.exists():
            return None
        return float((json.loads(p.read_text()).get("best") or {}).get("mpjpe_cm", float("nan")))
    f, n = best(f"modifiedA_esc_scr{sid}_full"), best(f"modifiedA_esc_scr{sid}_nofwd")
    if f is None and n is None:
        lines.append(f"| ESC-S{sid} | — | — | — | pending |"); continue
    gap = (n - f) if f is not None and n is not None and math.isfinite(f) and math.isfinite(n) else None
    gs = f"{gap:+.3f}" if gap is not None else "—"
    fs = f"{f:.3f}" if f is not None and math.isfinite(f) else "—"
    ns = f"{n:.3f}" if n is not None and math.isfinite(n) else "—"
    lines.append(f"| ESC-S{sid} | {fs} | {ns} | {gs} | {'done' if gap is not None else 'partial'} |")
    if gap is not None and gap > best_g:
        best_g, best_s = gap, sid
if best_s is not None:
    lines += ["", f"**Best ESC screen:** S{best_s} gap={best_g:+.3f} cm"]
md = Path("log/MODIFIED_A_SCREEN.md")
base = md.read_text() if md.exists() else "# modified_A screen results\n"
# replace prior escalation section if present
if "## Escalation" in base:
    base = base.split("## Escalation")[0].rstrip() + "\n"
md.write_text(base + "\n".join(lines) + "\n")
print(md.read_text())
PY
}

promote_and_full() {
  local sid="$1"
  echo "[esc] promoting ESC-S${sid} -> modified_a_esc_best.yml"
  cp "config/modified_a_esc_scr${sid}.yml" "config/modified_a_esc_best.yml"
  cp "config/modified_a_esc_scr${sid}_nofwd.yml" "config/modified_a_esc_best_nofwd.yml"
  "$PYTHON" - <<'PY'
from pathlib import Path
import re
for name in ["config/modified_a_esc_best.yml", "config/modified_a_esc_best_nofwd.yml"]:
    p = Path(name)
    t = p.read_text()
    t = re.sub(r"(?m)^max_epochs:\s*\d+", "max_epochs: 200", t)
    t = re.sub(r"(?m)^early_stop_patience:\s*\d+", "early_stop_patience: 25", t)
    if "promoted from ESC screen" not in t:
        t = "# promoted from ESC screen for full-budget train\n" + t
    p.write_text(t)
print("wrote esc best configs")
PY
  run_full() {
    local exp="$1" cfg="$2"
    if [[ -f "weights/${exp}/summary.json" ]]; then echo "SKIP finished $exp"; return 0; fi
    if [[ -f "log/${exp}.pid" ]] && kill -0 "$(cat "log/${exp}.pid")" 2>/dev/null; then
      echo "SKIP running $exp"; return 0
    fi
    # if mid-run best.pt without summary from prior S6 promote, use _v2
    if [[ -f "weights/${exp}/best.pt" && ! -f "weights/${exp}/summary.json" ]]; then
      echo "WARN $exp best.pt exists without complete summary; using ${exp}_v2"
      exp="${exp}_v2"
    fi
    if [[ -f "weights/${exp}/summary.json" ]]; then echo "SKIP finished $exp"; return 0; fi
    setsid nohup env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" -u main.py \
      --model modified_A --exp-id "$exp" --config "$cfg" --split-json "$SPLIT" --device cuda:0 \
      >"log/${exp}.log" 2>&1 &
    echo $! >"log/${exp}.pid"
    echo "Started $exp pid=$(cat log/${exp}.pid)"
    echo "$exp" > "log/${exp}.actual_exp_id"
  }
  # Prefer new ids so we never clobber stopped S6 promote artifacts
  run_full modifiedA_esc_full config/modified_a_esc_best.yml
  run_full modifiedA_esc_nofwd_full config/modified_a_esc_best_nofwd.yml
}

for sid in 0 1 3; do
  bash scripts/run_modified_a_esc_screens.sh "S${sid}"
  wait_pair "$sid"
  append_esc_md
  read -r BEST_SID BEST_GAP <<<"$(best_gap_sid)"
  CUR=$("$PYTHON" - <<PY
import json, math
from pathlib import Path
def best(exp):
    p = Path(f"weights/{exp}/summary.json")
    if not p.exists(): return None
    return float((json.loads(p.read_text()).get("best") or {}).get("mpjpe_cm", float("nan")))
f, n = best("modifiedA_esc_scr${sid}_full"), best("modifiedA_esc_scr${sid}_nofwd")
print((n - f) if f is not None and n is not None else -999)
PY
)
  echo "[esc] after S${sid}: best_sid=${BEST_SID} best_gap=${BEST_GAP} cur=${CUR}"
  awk -v g="$BEST_GAP" -v t="$GAP_TARGET" 'BEGIN{exit !(g+0>=t+0)}' && {
    echo "[esc] early stop: gap ${BEST_GAP} >= ${GAP_TARGET}"; break
  }
done

read -r BEST_SID BEST_GAP <<<"$(best_gap_sid)"
echo "[esc] FINAL best ESC-S${BEST_SID} gap=${BEST_GAP}"
if [[ "$BEST_SID" -lt 0 ]]; then BEST_SID=0; fi
promote_and_full "$BEST_SID"

wait_full() {
  local exp="$1"
  # resolve actual exp id if redirected to _v2
  if [[ -f "log/${exp}.actual_exp_id" ]]; then exp=$(cat "log/${exp}.actual_exp_id"); fi
  local pid; pid=$(cat "log/${exp}.pid" 2>/dev/null || true)
  echo "[esc] waiting $exp"
  while true; do
    if [[ -f "weights/${exp}/summary.json" ]]; then
      [[ -z "${pid:-}" ]] || ! kill -0 "$pid" 2>/dev/null && break
    fi
    [[ -n "${pid:-}" ]] && ! kill -0 "$pid" 2>/dev/null && break
    sleep 120
  done
}
wait_full modifiedA_esc_full
wait_full modifiedA_esc_nofwd_full

# Also finish incomplete S6 promote summaries already snapshotted; eval esc + prior
"$PYTHON" -u scripts/eval_official_testset.py \
  --only modifiedA_esc_full modifiedA_esc_nofwd_full modifiedA_full modifiedA_nofwd_full --merge || true

append_esc_md

# Gesture: baselines + A–D + esc modifiedA if ready
BACKBONES=(hand3d mediapipe nanotapor A B C D)
if [[ -f weights/modifiedA_esc_full/best.pt ]]; then
  # train_gesture maps modified_A -> modifiedA_full; add esc via pose exp alias
  BACKBONES+=(modified_A)
fi
# Patch: also try esc exp by temporarily linking or extending POSE_EXPS — train via --backbones modifiedA_esc_full after adding to POSE_EXPS
echo "[esc] launching gesture"
setsid nohup env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" -u scripts/train_gesture.py \
  --config config/gesture.yml --split-json data/tapor/gesture_split.json --device cuda:0 \
  --backbones hand3d mediapipe nanotapor A B C D \
  --out-md log/GESTURE_RESULTS.md \
  >log/gesture_batch_missing.log 2>&1 &
echo $! > log/gesture_batch_missing.pid
gpid=$(cat log/gesture_batch_missing.pid)
while kill -0 "$gpid" 2>/dev/null; do sleep 60; done

# modifiedA gesture if esc full exists (use exp id override)
if [[ -f weights/modifiedA_esc_full/best.pt ]]; then
  "$PYTHON" -u scripts/train_gesture.py --config config/gesture.yml \
    --split-json data/tapor/gesture_split.json --device cuda:0 \
    --backbones modifiedA_esc_full --out-md log/GESTURE_RESULTS.md \
    >>log/gesture_batch_missing.log 2>&1 || true
fi

echo done > log/modified_a_esc_orch.done
echo "[esc] ALL DONE"

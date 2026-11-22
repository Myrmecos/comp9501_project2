#!/usr/bin/env bash
# After micro OK: labeled-frac ablation for modified_B/C/D (and report modified_A/P4).
# Usage: bash small_ws/scripts/run_modified_bcd_frac.sh
# Only launches models whose micro gap >= 0 (Forward not clearly worse), or all if FORCE=1.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SW="$ROOT/small_ws"
PC="$ROOT/physics_correct_ws"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset PYTHONPATH || true
FORCE="${FORCE:-0}"

active_exps() {
  local n
  n=$(ps aux | grep -E 'main.py .*--exp-id' | grep -v grep | grep -oE -- '--exp-id [^ ]+' | awk '{print $2}' | sort -u | wc -l || true)
  echo "${n:-0}"
}
wait_slots() {
  local max="${1:-2}"
  while true; do
    local n; n=$(active_exps)
    echo "[fracBCD] active_exps=$n (max=$max)"
    [[ "$n" -le "$max" ]] && break
    sleep 90
  done
}

# Build fXX configs from micro P4 recipe
"$PYTHON" - <<'PY'
import copy, yaml
from pathlib import Path
SW=Path("small_ws/config")
budgets={
  "05": dict(max_epochs=80, early_stop_patience=15, phase1_epochs=25),
  "10": dict(max_epochs=70, early_stop_patience=14, phase1_epochs=22),
  "20": dict(max_epochs=60, early_stop_patience=12, phase1_epochs=20),
  "60": dict(max_epochs=100, early_stop_patience=18, phase1_epochs=30),
}
for letter, model in [("B","modified_B"),("C","modified_C"),("D","modified_D")]:
    base=yaml.safe_load((SW/f"M{letter}.yml").read_text())
    nofwd=yaml.safe_load((SW/f"M{letter}_nofwd.yml").read_text())
    for frac, bud in budgets.items():
        for tag, src in [(f"M{letter}_f{frac}", base), (f"M{letter}_f{frac}_nofwd", nofwd)]:
            c=copy.deepcopy(src); c.update(bud); c["phypose_model"]=model
            (SW/f"{tag}.yml").write_text(f"# {tag}\n"+yaml.safe_dump(c, sort_keys=False))
print("wrote frac configs")
PY

micro_ok() {
  local id="$1"
  "$PYTHON" - <<PY
import json, math
from pathlib import Path
def mpj(e):
    p=Path("small_ws/weights")/e/"summary.json"
    if not p.exists(): return float("nan")
    return float((json.loads(p.read_text()).get("best") or {}).get("mpjpe_cm", float("nan")))
f,n=mpj("$id"), mpj("${id}_nofwd")
gap=(n-f) if math.isfinite(f) and math.isfinite(n) else float("nan")
print(gap)
import sys
# Include near-ties / mild negatives (e.g. modified_C micro -0.03) so we still get % curves.
sys.exit(0 if ($FORCE==1) or (math.isfinite(gap) and gap>=-0.05) else 1)
PY
}

run_one() {
  local art="$1" exp="$2" cfg="$3" split="$4" model="$5"
  if [[ -f "$art/weights/${exp}/summary.json" ]]; then echo "SKIP $exp"; return 0; fi
  if [[ -f "$art/log/${exp}.pid" ]] && kill -0 "$(cat "$art/log/${exp}.pid")" 2>/dev/null; then
    echo "SKIP running $exp"; return 0
  fi
  mkdir -p "$art/log" "$art/weights"
  echo "[fracBCD] start $exp"
  setsid nohup env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" -u main.py \
    --model "$model" --exp-id "$exp" --config "$cfg" --split-json "$split" \
    --device cuda:0 --artifact-root "$art" >"$art/log/${exp}.log" 2>&1 &
  echo $! >"$art/log/${exp}.pid"
}

wait_exp() {
  local art="$1" exp="$2" pid
  pid=$(cat "$art/log/${exp}.pid" 2>/dev/null || true)
  while true; do
    [[ -f "$art/weights/${exp}/summary.json" ]] && break
    [[ -n "${pid:-}" ]] && ! kill -0 "$pid" 2>/dev/null && break
    sleep 60
  done
}

collect_all() {
  "$PYTHON" "$SW/scripts/collect_modified_bcd_frac.py"
}

for letter in B C D; do
  id="M${letter}"
  model="modified_${letter}"
  if ! micro_ok "$id"; then
    echo "[fracBCD] skip $id (micro gap bad); set FORCE=1 to override"
    continue
  fi
  for frac in 05 10 20 60; do
    wait_slots 4
    split="$ROOT/data/tapor/limited_label_f${frac}_coupled.json"
    exp="M${letter}_f${frac}"
    run_one "$SW" "$exp" "$SW/config/${exp}.yml" "$split" "$model"
    run_one "$SW" "${exp}_nofwd" "$SW/config/${exp}_nofwd.yml" "$split" "$model"
    wait_exp "$SW" "$exp"
    wait_exp "$SW" "${exp}_nofwd"
    collect_all || true
  done
done
collect_all
echo "[fracBCD] done"

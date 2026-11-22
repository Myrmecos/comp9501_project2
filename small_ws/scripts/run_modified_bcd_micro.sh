#!/usr/bin/env bash
# Micro screen: modified_B/C/D with P4 recipe vs nofwd.
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

active_exps() {
  local n
  n=$(ps aux | grep -E 'main.py .*--exp-id' | grep -v grep | grep -oE -- '--exp-id [^ ]+' | awk '{print $2}' | sort -u | wc -l || true)
  echo "${n:-0}"
}

wait_slots() {
  local max="${1:-2}"
  while true; do
    local n; n=$(active_exps)
    echo "[MBCD] active_exps=$n (max=$max)"
    [[ "$n" -le "$max" ]] && break
    sleep 60
  done
}

run_one() {
  local exp="$1" cfg="$2"
  local model
  model=$(grep -E '^phypose_model:' "$cfg" | awk '{print $2}' | tr -d '"' | tr -d "'")
  model=${model:-modified_A}
  if [[ -f "$SW/weights/${exp}/summary.json" ]]; then echo "SKIP finished $exp"; return 0; fi
  if [[ -f "$SW/log/${exp}.pid" ]] && kill -0 "$(cat "$SW/log/${exp}.pid")" 2>/dev/null; then
    echo "SKIP running $exp"; return 0
  fi
  echo "[MBCD] start $exp model=$model"
  setsid nohup env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" -u main.py \
    --model "$model" --exp-id "$exp" --config "$cfg" --split-json "$SPLIT" \
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
    sleep 45
  done
}

collect() {
  "$PYTHON" - <<'PY'
import json, math
from pathlib import Path
SW=Path("small_ws")
pairs=[("MB","modified_B"),("MC","modified_C"),("MD","modified_D")]
lines=["# modified_B/C/D micro screen (P4 recipe)","",
       "gap = nofwd − Forward. residual OFF, λ_recon 0.02→0, keep_depth_attached, mask/inv-skip.","",
       "| ID | model | Forward | nofwd | gap | status |",
       "|----|-------|---------|-------|-----|--------|"]
for cid, model in pairs:
    def best(e):
        p=SW/"weights"/e/"summary.json"
        if not p.exists(): return None
        return float((json.loads(p.read_text()).get("best") or {}).get("mpjpe_cm", float("nan")))
    f,n=best(cid),best(cid+"_nofwd")
    if f is None and n is None:
        lines.append(f"| {cid} | {model} | — | — | — | pending |")
    else:
        gap=(n-f) if f is not None and n is not None and math.isfinite(f) and math.isfinite(n) else None
        fs=f"{f:.3f}" if f is not None and math.isfinite(f) else "—"
        ns=f"{n:.3f}" if n is not None and math.isfinite(n) else "—"
        gs=f"{gap:+.3f}" if gap is not None else "—"
        st="done" if gap is not None else "partial"
        lines.append(f"| {cid} | {model} | {fs} | {ns} | {gs} | {st} |")
out=SW/"log"/"MODIFIED_BCD_MICRO.md"
out.write_text("\n".join(lines)+"\n")
print(out.read_text())
PY
}

for id in MB MC MD; do
  wait_slots 4
  run_one "$id" "$SW/config/${id}.yml"
  run_one "${id}_nofwd" "$SW/config/${id}_nofwd.yml"
  wait_exp "$id"
  wait_exp "${id}_nofwd"
  collect
done
collect
echo "[MBCD] micro done"

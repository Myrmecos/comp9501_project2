#!/usr/bin/env bash
# After C6_full finishes: if gap < target, try C8_full, C12_full, then C13–C16→f20→full.
# Assumes promote_until_win may have soft-failed. Never touches hybridA_*.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SW="$ROOT/small_ws"
cd "$ROOT"
PYTHON="/home/liuxuan/anaconda3/envs/py3.10/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset PYTHONPATH || true
GAP_TARGET=0.05
FULL_SPLIT="$ROOT/data/tapor/full_split.json"
F20_SPLIT="$ROOT/data/tapor/limited_label_f20_coupled.json"
ART="$SW"

model_from_cfg() {
  local m; m=$(grep -E '^phypose_model:' "$1" 2>/dev/null | awk '{print $2}' | tr -d '"' | tr -d "'")
  echo "${m:-modified_A}"
}

run_one() {
  local exp="$1" cfg="$2" split="$3"
  local model; model=$(model_from_cfg "$cfg")
  if [[ -f "$SW/weights/${exp}/summary.json" ]]; then echo "SKIP finished $exp"; return 0; fi
  if [[ -f "$SW/log/${exp}.pid" ]] && kill -0 "$(cat "$SW/log/${exp}.pid")" 2>/dev/null; then
    echo "SKIP running $exp"; return 0
  fi
  echo "[cont] start $exp"
  setsid nohup env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" -u main.py \
    --model "$model" --exp-id "$exp" --config "$cfg" --split-json "$split" \
    --device cuda:0 --artifact-root "$ART" >"$SW/log/${exp}.log" 2>&1 &
  echo $! >"$SW/log/${exp}.pid"
}

wait_exp() {
  local exp="$1" pid; pid=$(cat "$SW/log/${exp}.pid" 2>/dev/null || true)
  while true; do
    [[ -f "$SW/weights/${exp}/summary.json" ]] && break
    [[ -n "${pid:-}" ]] && ! kill -0 "$pid" 2>/dev/null && break
    sleep 60
  done
}

gap_of() {
  "$PYTHON" - <<PY
import json, math
from pathlib import Path
def mpjpe(exp):
    p = Path("$SW")/"weights"/exp/"summary.json"
    if not p.exists(): return float("nan")
    return float((json.loads(p.read_text()).get("best") or {}).get("mpjpe_cm", float("nan")))
f, n = mpjpe("$1"), mpjpe("$2")
gap = (n - f) if math.isfinite(f) and math.isfinite(n) else float("nan")
import sys
print(f"[cont] $1={f:.4f} $2={n:.4f} gap={gap:+.4f}", file=sys.stderr, flush=True)
print(gap)
PY
}

refresh_md() {
  "$PYTHON" - <<'PY'
import json, math
from pathlib import Path
SW=Path("small_ws")
ids=["C6_f20","C8_f20","C12_f20","C6_full","C8_full","C12_full",
     "C13_f20","C14_f20","C15_f20","C16_f20","C13_full","C14_full","C15_full","C16_full"]
lines=["# small_ws promotion (Forward vs nofwd)","","gap = nofwd − full (positive ⇒ Forward helps).","",
       "| ID | full MPJPE | nofwd MPJPE | gap | status |",
       "|----|------------|-------------|-----|--------|"]
best_g, best_id = -1e9, None
for cid in ids:
    def best(exp):
        p=SW/"weights"/exp/"summary.json"
        if not p.exists(): return None
        return float((json.loads(p.read_text()).get("best") or {}).get("mpjpe_cm", float("nan")))
    f,n=best(cid),best(cid+"_nofwd")
    if f is None and n is None:
        # skip totally missing
        continue
    gap=(n-f) if f is not None and n is not None and math.isfinite(f) and math.isfinite(n) else None
    fs=f"{f:.3f}" if f is not None and math.isfinite(f) else "—"
    ns=f"{n:.3f}" if n is not None and math.isfinite(n) else "—"
    gs=f"{gap:+.3f}" if gap is not None else "—"
    st="done" if gap is not None else "partial"
    lines.append(f"| {cid} | {fs} | {ns} | {gs} | {st} |")
    if gap is not None and gap>best_g: best_g,best_id=gap,cid
lines+=["", f"**Best:** {best_id} gap={best_g:+.3f} cm" if best_id else ""]
if best_id and best_g>=0.05:
    lines.append("Forward **beats** nofwd on this scale.")
(SW/"log"/"PROMOTE_SCREEN.md").write_text("\n".join(lines)+"\n")
print((SW/"log"/"PROMOTE_SCREEN.md").read_text())
PY
}

run_pair() {
  local tag="$1" cfg="$2" cfg_n="$3" split="$4"
  [[ -f "$cfg" ]] || { echo "missing $cfg"; return 1; }
  run_one "$tag" "$cfg" "$split"
  run_one "${tag}_nofwd" "$cfg_n" "$split"
  wait_exp "$tag"; wait_exp "${tag}_nofwd"
  refresh_md
  local g; g=$(gap_of "$tag" "${tag}_nofwd" | tail -1)
  echo "[cont] pair $tag gap=$g"
  awk -v g="$g" -v t="$GAP_TARGET" 'BEGIN{exit !(g+0>=t+0)}'
}

# Wait for in-flight C6_full if still running
if [[ -f "$SW/log/C6_full.pid" ]] && kill -0 "$(cat "$SW/log/C6_full.pid")" 2>/dev/null; then
  echo "[cont] waiting in-flight C6_full"
  wait_exp C6_full
fi
if [[ -f "$SW/log/C6_full_nofwd.pid" ]] && kill -0 "$(cat "$SW/log/C6_full_nofwd.pid")" 2>/dev/null; then
  wait_exp C6_full_nofwd
fi
refresh_md

if [[ -f "$SW/weights/C6_full/summary.json" && -f "$SW/weights/C6_full_nofwd/summary.json" ]]; then
  g=$(gap_of C6_full C6_full_nofwd | tail -1)
  if awk -v g="$g" -v t="$GAP_TARGET" 'BEGIN{exit !(g+0>=t+0)}'; then
    echo "[cont] SUCCESS C6_full gap=$g" | tee "$SW/log/PROMOTE_WIN.txt"
    echo success >> "$SW/log/PROMOTE_WIN.txt"
    echo DONE > "$SW/log/promote_orch.done"
    exit 0
  fi
  echo "[cont] C6_full gap=$g < $GAP_TARGET — try C8_full / C12_full"
fi

for tag in C8_full C12_full; do
  if run_pair "$tag" "$SW/config/${tag}.yml" "$SW/config/${tag}_nofwd.yml" "$FULL_SPLIT"; then
    echo "[cont] SUCCESS $tag" | tee "$SW/log/PROMOTE_WIN.txt"
    echo success >> "$SW/log/PROMOTE_WIN.txt"
    echo DONE > "$SW/log/promote_orch.done"
    exit 0
  fi
done

# Micro→f20→full for C13–C16
for tag in C13 C14 C15 C16; do
  [[ -f "$SW/config/${tag}.yml" ]] || continue
  echo "[cont] micro screen $tag"
  run_one "$tag" "$SW/config/${tag}.yml" "$SW/data/micro_split.json"
  run_one "${tag}_nofwd" "$SW/config/${tag}_nofwd.yml" "$SW/data/micro_split.json"
  wait_exp "$tag"; wait_exp "${tag}_nofwd"
  "$PYTHON" "$SW/scripts/collect_micro.py" >/dev/null || true
  g=$(gap_of "$tag" "${tag}_nofwd" | tail -1)
  echo "[cont] micro $tag gap=$g"
  awk -v g="$g" -v t="$GAP_TARGET" 'BEGIN{exit !(g+0>=t+0)}' || continue

  "$PYTHON" - <<PY
import copy, yaml, re
from pathlib import Path
def load(p):
    t=Path(p).read_text(); t=re.sub(r'^#.*\n','',t); return yaml.safe_load(t)
def dump(p,c,msg):
    Path(p).write_text(f'# {msg}\n'+yaml.safe_dump(c, sort_keys=False, default_flow_style=False))
cfg=load(f"small_ws/config/${tag}.yml")
for suffix, ep, pat, p1 in [("f20",60,12,20),("full",120,20,30)]:
    c=copy.deepcopy(cfg); c['max_epochs']=ep; c['early_stop_patience']=pat; c['phase1_epochs']=p1
    nf=copy.deepcopy(c); nf['use_forward']=False; nf['lambda_recon']=0; nf['phase1_lambda_recon']=0; nf['phase2_lambda_recon']=0
    dump(f"small_ws/config/${tag}_{suffix}.yml", c, f"${tag} {suffix}")
    dump(f"small_ws/config/${tag}_{suffix}_nofwd.yml", nf, f"${tag} {suffix} nofwd")
print("wrote f20/full for ${tag}")
PY
  if run_pair "${tag}_f20" "$SW/config/${tag}_f20.yml" "$SW/config/${tag}_f20_nofwd.yml" "$F20_SPLIT"; then
    if run_pair "${tag}_full" "$SW/config/${tag}_full.yml" "$SW/config/${tag}_full_nofwd.yml" "$FULL_SPLIT"; then
      echo "[cont] SUCCESS ${tag}_full" | tee "$SW/log/PROMOTE_WIN.txt"
      echo success >> "$SW/log/PROMOTE_WIN.txt"
      echo DONE > "$SW/log/promote_orch.done"
      exit 0
    fi
  fi
done

echo "[cont] exhausted candidates" | tee "$SW/log/PROMOTE_FAIL.txt"
exit 1

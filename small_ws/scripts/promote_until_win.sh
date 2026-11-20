#!/usr/bin/env bash
# Promote promising micro winners (C6/C8/C12) until Forward beats nofwd on larger data.
# Never touches weights/hybridA_*. Artifacts stay under small_ws/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SW="$ROOT/small_ws"
cd "$ROOT"
PYTHON="/home/liuxuan/anaconda3/envs/py3.10/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset PYTHONPATH || true
mkdir -p "$SW/log" "$SW/weights"

GAP_TARGET=0.05
F20_SPLIT="$ROOT/data/tapor/limited_label_f20_coupled.json"
FULL_SPLIT="$ROOT/data/tapor/full_split.json"
ART="$SW"

model_from_cfg() {
  local m
  m=$(grep -E '^phypose_model:' "$1" 2>/dev/null | awk '{print $2}' | tr -d '"' | tr -d "'")
  echo "${m:-modified_A}"
}

run_one() {
  local exp="$1" cfg="$2" split="$3"
  local model
  model=$(model_from_cfg "$cfg")
  if [[ -f "$SW/log/${exp}.pid" ]] && kill -0 "$(cat "$SW/log/${exp}.pid")" 2>/dev/null; then
    echo "SKIP running $exp"; return 0
  fi
  if [[ -f "$SW/weights/${exp}/summary.json" ]]; then
    echo "SKIP finished $exp"; return 0
  fi
  if [[ -f "$SW/weights/${exp}/best.pt" && ! -f "$SW/weights/${exp}/summary.json" ]]; then
    echo "WARN $exp has best.pt w/o summary — skip to avoid clobber"; return 0
  fi
  echo "[promote] start $exp model=$model split=$(basename "$split")"
  setsid nohup env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" -u main.py \
    --model "$model" --exp-id "$exp" --config "$cfg" --split-json "$split" \
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

gap_of() {
  local full="$1" nofwd="$2"
  "$PYTHON" - <<PY
import json, math
from pathlib import Path
def mpjpe(exp):
    p = Path("$SW")/"weights"/exp/"summary.json"
    if not p.exists():
        return float("nan")
    return float((json.loads(p.read_text()).get("best") or {}).get("mpjpe_cm", float("nan")))
f, n = mpjpe("$full"), mpjpe("$nofwd")
gap = (n - f) if math.isfinite(f) and math.isfinite(n) else float("nan")
import sys
print(f"[promote] $full={f:.4f} $nofwd={n:.4f} gap={gap:+.4f}", file=sys.stderr, flush=True)
print(gap)
PY
}

append_promote_md() {
  "$PYTHON" - <<'PY'
import json, math
from pathlib import Path
SW = Path("small_ws")
pairs = [
    ("C6_f20", "C6_f20", "C6_f20_nofwd"),
    ("C8_f20", "C8_f20", "C8_f20_nofwd"),
    ("C12_f20", "C12_f20", "C12_f20_nofwd"),
    ("C6_full", "C6_full", "C6_full_nofwd"),
    ("C8_full", "C8_full", "C8_full_nofwd"),
    ("C12_full", "C12_full", "C12_full_nofwd"),
]
lines = [
    "# small_ws promotion (Forward vs nofwd)",
    "",
    "gap = nofwd − full (positive ⇒ Forward helps). Original HybridA untouched.",
    "",
    "| ID | full MPJPE | nofwd MPJPE | gap | status |",
    "|----|------------|-------------|-----|--------|",
]
best_g, best_id = -1e9, None
for cid, fe, ne in pairs:
    def best(exp):
        p = SW/"weights"/exp/"summary.json"
        if not p.exists(): return None
        return float((json.loads(p.read_text()).get("best") or {}).get("mpjpe_cm", float("nan")))
    f, n = best(fe), best(ne)
    if f is None and n is None:
        lines.append(f"| {cid} | — | — | — | pending |"); continue
    gap = (n-f) if f is not None and n is not None and math.isfinite(f) and math.isfinite(n) else None
    fs = f"{f:.3f}" if f is not None and math.isfinite(f) else "—"
    ns = f"{n:.3f}" if n is not None and math.isfinite(n) else "—"
    gs = f"{gap:+.3f}" if gap is not None else "—"
    st = "done" if gap is not None else "partial"
    lines.append(f"| {cid} | {fs} | {ns} | {gs} | {st} |")
    if gap is not None and gap > best_g:
        best_g, best_id = gap, cid
lines.append("")
if best_id is not None:
    lines.append(f"**Best promote:** {best_id} gap={best_g:+.3f} cm")
    if best_g >= 0.05:
        lines.append("Forward **beats** nofwd on this scale.")
(SW/"log"/"PROMOTE_SCREEN.md").write_text("\n".join(lines)+"\n")
print((SW/"log"/"PROMOTE_SCREEN.md").read_text())
PY
}

run_pair() {
  local tag="$1" cfg="$2" cfg_n="$3" split="$4"
  run_one "$tag" "$cfg" "$split"
  run_one "${tag}_nofwd" "$cfg_n" "$split"
  wait_exp "$tag"
  wait_exp "${tag}_nofwd"
  append_promote_md
  local g
  g=$(gap_of "$tag" "${tag}_nofwd" | head -1)
  echo "[promote] pair $tag gap=$g"
  awk -v g="$g" -v t="$GAP_TARGET" 'BEGIN{exit !(g+0>=t+0)}'
}

echo "[promote] === f20 stage ==="
WINNER=""
for tag in C6_f20 C8_f20 C12_f20; do
  base="${tag%_f20}"
  if run_pair "$tag" "$SW/config/${tag}.yml" "$SW/config/${tag}_nofwd.yml" "$F20_SPLIT"; then
    echo "[promote] f20 WIN: $tag"
    WINNER="$tag"
    break
  fi
done

if [[ -z "$WINNER" ]]; then
  echo "[promote] no f20 win ≥ $GAP_TARGET — iterating micro variants C13–C16"
  # C13–C16 generated by companion python if missing; run if configs exist
  for tag in C13 C14 C15 C16; do
    [[ -f "$SW/config/${tag}.yml" ]] || continue
    run_one "$tag" "$SW/config/${tag}.yml" "$SW/data/micro_split.json"
    run_one "${tag}_nofwd" "$SW/config/${tag}_nofwd.yml" "$SW/data/micro_split.json"
    wait_exp "$tag"; wait_exp "${tag}_nofwd"
    "$PYTHON" "$SW/scripts/collect_micro.py" >/dev/null || true
    g=$(gap_of "$tag" "${tag}_nofwd" | head -1)
    echo "[promote] micro $tag gap=$g"
    if awk -v g="$g" -v t="$GAP_TARGET" 'BEGIN{exit !(g+0>=t+0)}'; then
      # promote this tag to f20 on the fly
      echo "[promote] promoting $tag to f20"
      "$PYTHON" - <<PY
import copy, yaml, re
from pathlib import Path
def load(p):
    t=Path(p).read_text(); t=re.sub(r'^#.*\n','',t); return yaml.safe_load(t)
def dump(p,c,msg):
    Path(p).write_text(f'# {msg}\n'+yaml.safe_dump(c, sort_keys=False, default_flow_style=False))
cfg=load(f"small_ws/config/${tag}.yml")
cfg['max_epochs']=60; cfg['early_stop_patience']=12; cfg['phase1_epochs']=20
nf=copy.deepcopy(cfg); nf['use_forward']=False; nf['lambda_recon']=0; nf['phase1_lambda_recon']=0; nf['phase2_lambda_recon']=0
dump(f"small_ws/config/${tag}_f20.yml", cfg, f"${tag} f20")
dump(f"small_ws/config/${tag}_f20_nofwd.yml", nf, f"${tag} f20 nofwd")
print("wrote f20 for ${tag}")
PY
      if run_pair "${tag}_f20" "$SW/config/${tag}_f20.yml" "$SW/config/${tag}_f20_nofwd.yml" "$F20_SPLIT"; then
        WINNER="${tag}_f20"
        break
      fi
    fi
  done
fi

if [[ -z "$WINNER" ]]; then
  echo "[promote] FAIL: no Forward>nofwd on f20 after iteration" | tee "$SW/log/PROMOTE_FAIL.txt"
  exit 1
fi

# Map winner to full configs
BASE="${WINNER%_f20}"
echo "[promote] === full_split stage for $BASE ==="
FULL_TAG="${BASE}_full"
run_pair "$FULL_TAG" "$SW/config/${FULL_TAG}.yml" "$SW/config/${FULL_TAG}_nofwd.yml" "$FULL_SPLIT" || {
  # if full configs missing, generate from f20 winner
  if [[ ! -f "$SW/config/${FULL_TAG}.yml" ]]; then
    "$PYTHON" - <<PY
import copy, yaml, re
from pathlib import Path
def load(p):
    t=Path(p).read_text(); t=re.sub(r'^#.*\n','',t); return yaml.safe_load(t)
cfg=load("small_ws/config/${WINNER}.yml")
cfg['max_epochs']=120; cfg['early_stop_patience']=20; cfg['phase1_epochs']=30
nf=copy.deepcopy(cfg); nf['use_forward']=False; nf['lambda_recon']=0; nf['phase1_lambda_recon']=0; nf['phase2_lambda_recon']=0
Path("small_ws/config/${FULL_TAG}.yml").write_text('# auto full\n'+yaml.safe_dump(cfg, sort_keys=False))
Path("small_ws/config/${FULL_TAG}_nofwd.yml").write_text('# auto full nofwd\n'+yaml.safe_dump(nf, sort_keys=False))
print("generated full cfgs")
PY
    run_pair "$FULL_TAG" "$SW/config/${FULL_TAG}.yml" "$SW/config/${FULL_TAG}_nofwd.yml" "$FULL_SPLIT"
  fi
}

g=$(gap_of "$FULL_TAG" "${FULL_TAG}_nofwd" | head -1)
append_promote_md
echo "$FULL_TAG gap=$g" > "$SW/log/PROMOTE_WIN.txt"
if awk -v g="$g" -v t="$GAP_TARGET" 'BEGIN{exit !(g+0>=t+0)}'; then
  echo "[promote] SUCCESS Forward beats nofwd on full_split: $FULL_TAG gap=$g"
  echo success >> "$SW/log/PROMOTE_WIN.txt"
else
  echo "[promote] full gap $g < $GAP_TARGET but continuing — record best effort"
  echo soft_fail >> "$SW/log/PROMOTE_WIN.txt"
fi
echo DONE > "$SW/log/promote_orch.done"

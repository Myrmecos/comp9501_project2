#!/usr/bin/env bash
# Read-only diagnostic for the remote TAPOR reproduction host.
set -euo pipefail

TAPOR_ROOT="${TAPOR_ROOT:-/workspace/TAPOR}"
DATA_ROOT="${DATA_ROOT:-/root/sj-tmp/tapor_data}"
RUNS_ROOT="${RUNS_ROOT:-/root/sj-tmp/tapor_runs}"
CHECKPOINT="${CHECKPOINT:-$TAPOR_ROOT/weights/tapor.pth}"
status=0

section() { printf '\n== %s ==\n' "$1"; }
ok() { printf '[OK] %s\n' "$1"; }
warn() { printf '[WARN] %s\n' "$1"; }
fail() { printf '[FAIL] %s\n' "$1" >&2; status=1; }

section "Platform"
if [[ "$(uname -s)" == "Linux" ]]; then ok "Linux: $(uname -sr)"; else fail "Expected Linux, got $(uname -s)"; fi
command -v python >/dev/null && ok "Python: $(python --version)" || fail "python is not on PATH"
command -v conda >/dev/null && { ok "Conda: $(conda --version)"; printf 'CONDA_DEFAULT_ENV=%s\n' "${CONDA_DEFAULT_ENV:-<none>}"; } || fail "conda is not on PATH"

section "GPU"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || fail "nvidia-smi query failed"
  nvidia-smi -L || fail "nvidia-smi -L failed"
  nvidia-smi -L | grep -qi 'RTX 4090' && ok "RTX 4090 visible" || warn "RTX 4090 string not found; confirm the assigned GPU"
else
  fail "nvidia-smi is not on PATH"
fi

section "Python packages and CUDA"
if command -v python >/dev/null; then
  python - <<'PY' || status=1
import importlib
import sys
required = ["torch", "torchvision", "cv2", "numpy", "timm", "tqdm", "tensorboard", "mano"]
failed = []
for name in required:
    try:
        module = importlib.import_module(name)
        print(f"[OK] import {name}: {getattr(module, '__version__', '<no __version__>')}")
    except Exception as exc:
        print(f"[FAIL] import {name}: {type(exc).__name__}: {exc}")
        failed.append(name)
try:
    import torch
    print(f"torch.version.cuda={torch.version.cuda}")
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"torch.cuda.device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.device_name={torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"[FAIL] CUDA probe: {type(exc).__name__}: {exc}")
    failed.append("torch CUDA")
sys.exit(bool(failed))
PY
fi

section "Paths and storage"
for path in "$TAPOR_ROOT" "$DATA_ROOT" "$RUNS_ROOT"; do
  if [[ -e "$path" ]]; then ok "exists: $path"; else warn "missing: $path"; fi
done
[[ -f "$TAPOR_ROOT/traintest.py" && -f "$TAPOR_ROOT/inference.py" ]] && ok "TAPOR source files found" || fail "TAPOR source incomplete: $TAPOR_ROOT"
[[ -d "$TAPOR_ROOT/Trainset" ]] && ok "Trainset link/directory present" || warn "Trainset is not linked yet"
[[ -d "$TAPOR_ROOT/Testset" ]] && ok "Testset link/directory present" || warn "Testset is not linked yet"
[[ -f "$CHECKPOINT" ]] && ok "checkpoint present: $CHECKPOINT" || warn "checkpoint missing: $CHECKPOINT"
df -h "$TAPOR_ROOT" "$DATA_ROOT" "$(dirname "$RUNS_ROOT")" 2>/dev/null || true

section "Official source revision"
if [[ -d "$TAPOR_ROOT/.git" ]]; then git -C "$TAPOR_ROOT" rev-parse HEAD && git -C "$TAPOR_ROOT" status --short; else warn "not a Git checkout: $TAPOR_ROOT"; fi

exit "$status"

#!/usr/bin/env bash
# 1) micro MB/MC/MD  2) if OK → labeled-frac ablation  3) collect report
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
bash "$ROOT/small_ws/scripts/run_modified_bcd_micro.sh"
bash "$ROOT/small_ws/scripts/run_modified_bcd_frac.sh"
python "$ROOT/small_ws/scripts/collect_modified_bcd_frac.py"
echo "[orch] modified B/C/D pipeline complete"

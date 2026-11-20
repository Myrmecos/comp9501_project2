# small_ws — fast Forward-vs-nofwd search

Isolated workspace so main `weights/` / `log/` stay clean.

## Goal

Find **parameter sets / structures** where `with Forward` beats `nofwd` on a **micro split** (2k train / 500 val frames from `full_split`), then promote winners to larger data.

**Score:** `gap = nofwd_mpjpe − full_mpjpe` (positive ⇒ Forward helps). Target gap ≥ **0.05 cm** on micro before scaling up.

## Layout

| path | role |
|------|------|
| `data/micro_split.json` | 2000 / 500 frame split |
| `config/` | candidate YAMLs (`*_nofwd.yml` pairs) |
| `weights/`, `log/` | all micro-run artifacts (`--artifact-root small_ws`) |
| `CANDIDATES.md` | hypothesis list |
| `log/MICRO_SCREEN.md` | live gap table |

## Run

```bash
# from repo root
bash small_ws/scripts/run_micro_candidates.sh C0 C1 C2   # paired full/nofwd
python small_ws/scripts/collect_micro.py
```

Uses repo `main.py` / `models/` with `--artifact-root small_ws`.

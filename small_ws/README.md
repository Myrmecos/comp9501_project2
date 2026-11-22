# small_ws — Forward vs nofwd ablation workspace

Isolated artifact root so main `weights/` / `log/` stay clean.
See the root [README.md](../README.md) for split definitions and the full ablation guide.

## Purpose

Screen and evaluate **modified_B / modified_C / modified_D** (and selected
modified_A recipes such as **C6**) where ForwardThermal beats the matched
`*_nofwd` ablation.

**Score:** `gap = nofwd_MPJPE − forward_MPJPE` (cm).

## Layout

| Path | Role |
|------|------|
| `data/micro_split.json` | 2k / 500 micro split |
| `config/` | Reported recipes: `C6*`, `MB*`, `MC*`, `MC2*`, `MD*` (+ `*_nofwd`) |
| `scripts/` | Micro / fraction runners and collectors |
| `weights/`, `log/` | Local artifacts (gitignored) |

## Quick start

```bash
# from repo root
bash small_ws/scripts/run_modified_bcd_micro.sh
bash small_ws/scripts/run_modified_bcd_frac.sh
# or full orchestrator:
bash small_ws/scripts/orch_modified_bcd.sh

python small_ws/scripts/collect_micro.py
python small_ws/scripts/collect_modified_bcd_frac.py
```

All training calls use `python -u main.py ... --artifact-root small_ws`.

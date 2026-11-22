# physics_correct_ws — pure-physics Forward (P4) workspace

Isolated artifact root for the **modified_A / P4** recipe: residual CNN off,
weak recon `0.02 → 0`, depth kept attached.

See the root [README.md](../README.md) for splits and the ablation protocol.

## Purpose

Measure whether a physics-correct Forward branch improves MPJPE vs the matched
nofwd ablation on micro and limited-label fractions (`f05`–`f60`).

**Score:** `gap = nofwd_MPJPE − forward_MPJPE` (cm).

## Layout

| Path | Role |
|------|------|
| `config/P4*.yml` | Micro, fraction, and full paired Forward / nofwd configs |
| `scripts/` | Micro runner, label-fraction sweep, collectors |
| `weights/`, `log/` | Local artifacts (gitignored) |

## Quick start

```bash
# from repo root
bash physics_correct_ws/scripts/run_micro.sh P4
python physics_correct_ws/scripts/collect_micro.py

bash physics_correct_ws/scripts/run_label_frac_sweep.sh
python physics_correct_ws/scripts/collect_label_frac.py
```

Training uses `python -u main.py ... --artifact-root physics_correct_ws`.

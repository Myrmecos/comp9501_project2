# COMP9501 Group 12 — Physics-Guided Thermal Hand Pose

This repository implements **thermal forward + inverse** hybrids for 3D hand-pose
estimation on the TAPOR dataset. The core idea is to estimate depth / thermal
primitives with an inverse network, then reconstruct the observed thermal map
with a differentiable **ForwardThermal** model and use reconstruction as an
extra training signal (especially useful when pose labels are scarce).

## Setup

```bash
cd comp9501_project2
conda activate <your-env>          # Python 3.10 + PyTorch recommended
pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH
```

Optional: point launchers at a specific interpreter with `export PYTHON=python3`.

### Data

Place TAPOR archives under `data/tapor/` (gitignored pickles):

| Split | Drive / source | Local path |
|-------|----------------|------------|
| Pose train | [Trainset](https://drive.google.com/file/d/12Ppoh15nomnVAKo2UFQRqgcLeDRh5hGk/view) | `data/tapor/Trainset/` |
| Pose test | [Testset](https://drive.google.com/file/d/1cnwdelIiV4V15rq3ANN93YsNvAjpvV5O/view) | `data/tapor/Testset/` |
| Gesture | [Gesture](https://drive.google.com/file/d/1JmTD0N_9hdhCS8JOSd9t2bQ_NmB9BbdB/view) | `data/tapor/gesture/` |

Related upstream: [TAPOR](https://github.com/aiot-lab/TAPOR), [TAP3D](https://github.com/aiot-lab/TAP3D).
See `data/tapor/MANIFEST.md` for local layout and schema notes.

Split JSON files under `data/tapor/*.json` are versioned so train/val protocols
are reproducible without uploading pickles.

## Repository layout

| Path | Role |
|------|------|
| `models/` | Hybrid A–D (+ `modified_*`), ForwardThermal, baselines |
| `config/` | Default full / nofwd / SSL / gesture / best modified_A configs |
| `scripts/` | Split builders, train launchers, eval, collectors |
| `scripts/tapor/` | Upstream TAPOR reproduction helpers |
| `small_ws/` | Isolated Forward-vs-nofwd screens + limited-label ablations for modified B/C/D |
| `physics_correct_ws/` | Pure-physics Forward recipe (**P4**) and label-fraction sweeps for modified A |
| `system_diagram/` | Architecture notes |
| `results/`, `figures/`, `results_presentation/` | Evaluation tables / plots |
| `unused/` | Archival experiments (gitignored; not required to reproduce) |

## Train / test splits

All split builders live in `scripts/make_*.py`. Regenerate from the repo root if
JSON files are missing.

### 1. Full supervised split — `full_split.json`

```bash
python scripts/make_full_split.py
```

- Uses the official TAPOR `traintest.py` train file list under `Trainset/`.
- Keeps frames with valid 21×3 joints (skips the first 20 frames per file).
- **80% / 10% / 10%** random shuffle (`seed=0`) → train / val / (held-out from train pool).
- Approx. counts: **26,751 train / 3,343 val**.

### 2. No-leak split — `full_split_noleak.json`

```bash
python scripts/make_noleak_split.py
```

- Same idea as full, with extra leakage checks vs. the official test set.
- Base pool for **limited-label** experiments: **27,652 train / 2,937 val**.

### 3. Limited-label (labeled fraction) splits

```bash
python scripts/make_limited_label_split.py
```

From `full_split_noleak.json` train frames, subsample a coupled (labeled) subset;
the remainder is unlabeled (thermal-only) for optional SSL pretrain.

| Tag | Labeled fraction | Unlabeled fraction | Coupled train frames | Unlabeled train frames |
|-----|------------------|--------------------|----------------------|------------------------|
| `f05` | 5% | 95% | 1,383 | 26,269 |
| `f10` | 10% | 90% | 2,766 | 24,886 |
| `f20` | 20% | 80% | 5,531 | 22,121 |
| `f40` | 40% | 60% | 11,061 | 16,591 |
| `f60` | 60% | 40% | 16,592 | 11,060 |

Emits per fraction:

- `limited_label_fXX_coupled.json` — supervised train (= labeled frames; **versioned**)
- `limited_label_fXX_ssl.json` — unlabeled train for optional SSL pretrain (local; regenerable)
- `limited_label_fXX_meta.json` — counts + audit (local; regenerable)

Validation stays the full no-leak val set (all labeled).

### 4. Micro split — `small_ws/data/micro_split.json`

Fast screening split: **2,000 train / 500 val** frames subsampled from the full
protocol. Used only for Forward-vs-nofwd architecture / hyperparameter search.

### 5. Smoke / gesture / official test

| File | Builder | Purpose |
|------|---------|---------|
| `small_split.json` | `make_small_split.py` | Tiny smoke train (file-level) |
| `gesture_split.json` | `make_gesture_split.py` | Gesture recognition |
| `official_test_split.json` | `make_official_test_split.py` | Official TAPOR test frames for final eval |

## Models

| CLI name | File | Notes |
|----------|------|-------|
| `A` / `B` / `C` / `D` | `models/hybrid_{a,b,c,d}.py` | Original hybrids |
| `modified_A` … `modified_D` | `models/hybrid_*_modified.py` | Depth-attach + KP fuse + physics-Forward knobs |
| `vanilla`, `mediapipe`, `nanotapor`, `hand3d`, `resnet` | baselines | Paper / course baselines |

ForwardThermal is toggled by config `use_forward` (default true). Ablation
configs set `use_forward: false` and zero recon weights (`config/full_nofwd.yml`
and every `*_nofwd.yml`).

## Training

### Full supervised (with Forward)

```bash
python -u main.py \
  --model A \
  --exp-id hybridA_full \
  --config config/full.yml \
  --split-json data/tapor/full_split.json \
  --device cuda:0
```

Same pattern for `B` / `C` / `D`. Checkpoints → `weights/<exp-id>/`, logs → `log/`.

### Forward vs no-Forward ablation (full data)

```bash
# Paired A–D without ForwardThermal
bash scripts/run_nofwd_ablation.sh

# Or manually:
python -u main.py \
  --model A \
  --exp-id hybridA_nofwd_full \
  --config config/full_nofwd.yml \
  --split-json data/tapor/full_split.json \
  --device cuda:0
```

**Score:** `gap = nofwd_MPJPE − forward_MPJPE` (cm). Positive ⇒ Forward helps.

### Limited-label Forward / nofwd ablation

1. Build splits (`make_limited_label_split.py`) if needed.
2. Train on `limited_label_fXX_coupled.json` with a Forward config and its
   matching `*_nofwd.yml`.
3. Reported recipes:

| Family | Workspace | Config stem | Split |
|--------|-----------|-------------|-------|
| Modified A (pure physics) | `physics_correct_ws/` | `P4_fXX` / `P4_fXX_nofwd` | `limited_label_fXX_coupled.json` |
| Modified B | `small_ws/` | `MB_fXX` / `MB_fXX_nofwd` | same |
| Modified C | `small_ws/` | `MC_fXX` / `MC_fXX_nofwd` | same |
| Modified C (residual on) | `small_ws/` | `MC2_fXX` / `MC2_fXX_nofwd` | same |
| Modified D | `small_ws/` | `MD_fXX` / `MD_fXX_nofwd` | same |

Example (20% labeled / 80% unlabeled, modified A / P4):

```bash
python -u main.py \
  --model modified_A \
  --exp-id P4_f20 \
  --config physics_correct_ws/config/P4_f20.yml \
  --split-json data/tapor/limited_label_f20_coupled.json \
  --artifact-root physics_correct_ws \
  --device cuda:0

python -u main.py \
  --model modified_A \
  --exp-id P4_f20_nofwd \
  --config physics_correct_ws/config/P4_f20_nofwd.yml \
  --split-json data/tapor/limited_label_f20_coupled.json \
  --artifact-root physics_correct_ws \
  --device cuda:0
```

Batch helpers:

```bash
# Micro screen then fraction sweep for modified B/C/D
bash small_ws/scripts/orch_modified_bcd.sh

# P4 label-fraction sweep (modified A)
bash physics_correct_ws/scripts/run_label_frac_sweep.sh
```

`--artifact-root` keeps experiment weights/logs inside `small_ws/` or
`physics_correct_ws/` so the main `weights/` tree stays clean.

### Micro Forward-vs-nofwd screen

```bash
bash small_ws/scripts/run_micro_candidates.sh C6
python small_ws/scripts/collect_micro.py

bash physics_correct_ws/scripts/run_micro.sh P4
python physics_correct_ws/scripts/collect_micro.py
```

## Evaluation

```bash
# All exps that have weights/*/best.pt under the default artifact root
python -u scripts/eval_official_testset.py \
  --split-json data/tapor/official_test_split.json \
  --device cuda:0

# One experiment (merge into the JSON report)
python -u scripts/eval_official_testset.py \
  --only hybridA_full A \
  --merge \
  --device cuda:0
```

Primary metric: **MPJPE (cm)**. Physics runs also log depth / hand-depth / recon MAE.

## Design notes (Forward useful)

Recipes that consistently beat nofwd (especially at mid labeled fractions):

- Prefer **weak / short** thermal recon (`lambda_recon` ≈ `0.02 → 0`) over strong recon.
- Optionally disable the residual CNN (`use_residual: false`) so recon trains physics.
- Keep depth attached into the pose path (`keep_depth_attached`) with depth keypoint fuse.

See `system_diagram/` and `proposal.txt` for the project motivation and diagrams.

## Citation / references

- Zhang et al., TAPOR (IMWUT 2025)
- Zhang et al., TAP3D (arXiv:2610.11241)

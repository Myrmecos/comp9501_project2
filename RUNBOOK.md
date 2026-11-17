# TAPOR Baseline Reproduction Runbook

This runbook reproduces the upstream TAPOR pose baseline only. Scripts are our orchestration layer; the network, losses, and train/test scripts remain the TAPOR authors' code in `/workspace/TAPOR`.

## Important Dataset Gate

The teammate-provided Google Drive ID `1JmTD0N_9hdhCS8JOSd9t2bQ_NmB9BbdB` is linked by the upstream README under **Gesture Recognition**, not under the main pose training/test dataset section. Do not train the pose baseline merely because the archive contains pickle files. The inspection command below must find all filenames hard-coded by `traintest.py` (123) and `inference.py` (207), with all required pose keys and valid labels. Otherwise, stop and obtain/confirm the correct pose dataset with the research group.

## Paths

The scripts default to these server paths:

```text
/workspace/TAPOR                         upstream source clone
/root/sj-tmp/tapor_data/Trainset         large official pose training pickles
/root/sj-tmp/tapor_data/Testset          large official pose test pickles
/root/sj-tmp/tapor_runs                  reports and console logs
/root/sj-tmp/conda-envs/tapor-official   main Conda environment
```

Do not place the large data, weights, logs, or Conda environment on the 30 GB system disk. The upstream scripts require `Trainset/`, `Testset/`, `weights/`, `Runs/`, `LogTrainTest/`, and `TestDataset_logs/` under `/workspace/TAPOR`; the scripts create only safe symbolic links for the two dataset directories.

## 1. Upload This Bundle

On the SuanJiaYun browser file manager:

1. Open the remote file manager and enter `/workspace`.
2. Use its Upload button (usually “Upload file” or the up-arrow icon) to select local `tapor-reproduction-scripts.zip`.
3. Confirm the upload finished and the displayed remote filename/size matches the local file.
4. In the web terminal, run:

```bash
cd /workspace
unzip -l tapor-reproduction-scripts.zip
rm -rf tapor-reproduction-scripts
unzip tapor-reproduction-scripts.zip -d tapor-reproduction-scripts
chmod +x tapor-reproduction-scripts/scripts/*.sh
```

`rm -rf` above removes only the explicitly named prior script bundle directory, not `/workspace/TAPOR`. Omit it if there is no old bundle or retain old bundles for comparison.

Set persistent variables in the current shell:

```bash
export REPRO_ROOT=/workspace/tapor-reproduction-scripts
export TAPOR_ROOT=/workspace/TAPOR
export DATA_ROOT=/root/sj-tmp/tapor_data
export RUNS_ROOT=/root/sj-tmp/tapor_runs
mkdir -p "$DATA_ROOT" "$RUNS_ROOT" /root/sj-tmp/conda-envs /root/sj-tmp/conda-pkgs /root/sj-tmp/incoming
```

## 2. Verify The Upstream Source

```bash
test -f "$TAPOR_ROOT/traintest.py"
test -f "$TAPOR_ROOT/inference.py"
git -C "$TAPOR_ROOT" rev-parse HEAD
```

The local audit reviewed commit `09b8e7c`; record the server commit, rather than assuming it matches.

## 3. Create The Official Main Environment

The official `environment.yml` targets Python 3.8, PyTorch 2.1, and CUDA 12.1. Use a path environment on the data disk. `CONDA_PKGS_DIRS` keeps the package cache off the system disk.

```bash
cd "$TAPOR_ROOT"
export CONDA_PKGS_DIRS=/root/sj-tmp/conda-pkgs
conda env create --prefix /root/sj-tmp/conda-envs/tapor-official --file environment.yml
conda activate /root/sj-tmp/conda-envs/tapor-official
python --version
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

Do this once. If solving fails because the original pinned environment is no longer solvable, save the solver output; do not silently substitute arbitrary PyTorch/CUDA versions. The optional `environment2.yml` is not required for this stage.

## 4. Upload And Extract Dataset / Weights

Use the browser file manager to upload the teammate archive and any supplied checkpoint into `/root/sj-tmp/incoming`. On the terminal, inspect archive contents before extraction:

```bash
cd /root/sj-tmp/incoming
ls -lh
unzip -l TEAM_DATASET.zip | sed -n '1,160p'
mkdir -p "$DATA_ROOT"
unzip TEAM_DATASET.zip -d "$DATA_ROOT/extracted"
find "$DATA_ROOT/extracted" -maxdepth 3 -type f -name '*.pkl' | sed -n '1,40p'
```

Do **not** assume the archive has the expected layout. After inspection, only create `Trainset` and `Testset` locations when the archive contains the corresponding official pose files. For example, if extraction yields separate directories with the verified files:

```bash
ln -s /root/sj-tmp/tapor_data/extracted/Trainset "$DATA_ROOT/Trainset"
ln -s /root/sj-tmp/tapor_data/extracted/Testset "$DATA_ROOT/Testset"
"$REPRO_ROOT/scripts/prepare_tapor_paths.sh"
cp /root/sj-tmp/incoming/tapor.pth "$TAPOR_ROOT/weights/tapor.pth"
```

Use `ln -s` only after checking the target path; do not replace an existing `Trainset`/`Testset` blindly. `prepare_tapor_paths.sh` links upstream `weights/`, `Runs/`, `LogTrainTest/`, and `TestDataset_logs/` to `/root/sj-tmp/tapor_runs/upstream_outputs/`; it refuses to replace an existing non-symlink upstream directory. Checkpoint filename must match the argument passed as `-wp`.

## 5. Validate Data And Environment

```bash
conda activate /root/sj-tmp/conda-envs/tapor-official
python "$REPRO_ROOT/scripts/inspect_dataset.py" --tapor-root "$TAPOR_ROOT" --data-root "$DATA_ROOT" --report "$RUNS_ROOT/dataset_inspection_report.json"
"$REPRO_ROOT/scripts/check_environment.sh"
```

The first inspection is intentionally non-blocking: it writes a report even when the archive is incomplete or is a gesture dataset. Read `official_train_missing`, `official_test_missing`, and `invalid_or_unreadable_official_pose_files` in the JSON report. The inspection opens one trusted pickle at a time and writes shape/dtype/range summaries, required-key checks, missing official filename lists, absent-label counts, depth-label diagnostics, and train/test overlap detection. The upstream lists share 11 basenames; when both split directories are available, the inspector hashes those files and rejects byte-identical train/test content. It does not train a model or modify data.

Only after the non-blocking report confirms the required pose protocol is complete, run the strict gate below. The later inference/training wrappers run this same strict gate automatically.

```bash
python "$REPRO_ROOT/scripts/inspect_dataset.py" --tapor-root "$TAPOR_ROOT" --data-root "$DATA_ROOT" --report "$RUNS_ROOT/dataset_inspection_strict.json" --require-complete
```

## 6. Apply The Minimal Training Compatibility Fix

The official `traintest.py:get_gpu_ids` discards valid `GPU-...` UUIDs parsed from normal `nvidia-smi -L` output, then indexes an empty list. This blocks training before model construction. Apply only the supplied separate patch:

```bash
"$REPRO_ROOT/scripts/apply_tapor_compat_patch.sh"
git -C "$TAPOR_ROOT" diff -- traintest.py
```

The patch changes GPU UUID parsing only. It does not change TAPOR architecture, checkpoints, data protocol, losses, optimizer, or metrics. It is idempotent.

## 7. Model Engineering Smoke Test

This uses a synthetic `(2, 1, 24, 32)` tensor to prove model construction, output shape `(2, 21, 3)`, finite output, and optionally strict checkpoint compatibility. It is **not** evaluation.

```bash
"$REPRO_ROOT/scripts/verify_model.py" --tapor-root "$TAPOR_ROOT" --device cuda --report "$RUNS_ROOT/model_smoke.json"
"$REPRO_ROOT/scripts/verify_model.py" --tapor-root "$TAPOR_ROOT" --device cuda --checkpoint "$TAPOR_ROOT/weights/tapor.pth" --report "$RUNS_ROOT/pretrained_model_check.json"
```

## 8. Run Official Pretrained Test Inference

```bash
"$REPRO_ROOT/scripts/run_pretrained_inference.sh" tapor.pth
```

It verifies complete official pose files and strict checkpoint loading, then calls exactly:

```bash
cd /workspace/TAPOR
python inference.py -m tapor -wp tapor.pth -fs 1 -ms 0 -hm 0 -fo 1 -mt 1 -v 0
```

Upstream writes `TestDataset_logs/`; our script preserves console logs and timing manifests in `/root/sj-tmp/tapor_runs/`. The official test loader uses `drop_last=True` per input file, which is intentionally preserved.

## 9. Run A One-Epoch Real-Data Pipeline Smoke Test

```bash
BATCH_SIZE=8 "$REPRO_ROOT/scripts/run_train_smoke.sh"
```

This runs the upstream training script for one epoch with real official training data and exact baseline architecture/loss settings, except batch size is 8 and epochs is 1. It verifies loading, forward/backward, optimizer update, validation, and checkpoint writing. It is not a benchmark reproduction.

## 10. Run The Full 200-Epoch Baseline

Run in tmux so a browser disconnect does not stop training:

```bash
tmux new -s tapor-baseline
conda activate /root/sj-tmp/conda-envs/tapor-official
export REPRO_ROOT=/workspace/tapor-reproduction-scripts TAPOR_ROOT=/workspace/TAPOR DATA_ROOT=/root/sj-tmp/tapor_data RUNS_ROOT=/root/sj-tmp/tapor_runs
"$REPRO_ROOT/scripts/run_train_full.sh"
```

Detach with `Ctrl-b d`; resume with `tmux attach -t tapor-baseline`. The default call is exactly the upstream README setting:

```bash
python traintest.py -m tapor -e 200 -b 48 -fs 1 -ms 0 -t 0 -lr 0.0001 -hm 0 -fo 1 -s 0 -mt 1 -ls jb -tqdm 1
```

Before reducing memory use, use the exact batch size 48. If 24 GB VRAM is insufficient, run `BATCH_SIZE=32 "$REPRO_ROOT/scripts/run_train_full.sh"`; the script writes a deviation record. TAPOR's supplied `-wp` mechanism loads model weights only: it does not restore optimizer state, scheduler state, original epoch, or the random split, so it is not a scientifically equivalent resumable training facility. Do not assume the run completes within three hours.

## 11. Evaluate A Newly Trained Checkpoint

Find the actual created checkpoint and run the same official test protocol:

```bash
ls -lht "$TAPOR_ROOT/weights"
"$REPRO_ROOT/scripts/evaluate_checkpoint.sh" ACTUAL_TIMESTAMP_tapor.pth
```

The script reports only metrics computed by upstream `inference.py`: MSE, MAE, shift/root drift, MPJPE, and PCK thresholds. Never copy synthetic smoke output into an experiment table.

## 12. Collect Results

```bash
"$REPRO_ROOT/scripts/collect_results.py" --tapor-root "$TAPOR_ROOT" --runs-root "$RUNS_ROOT" --output "$RUNS_ROOT/EXPERIMENT_SUMMARY.md"
sed -n '1,240p' "$RUNS_ROOT/EXPERIMENT_SUMMARY.md"
```

The collector uses actual `manifest.json`, dataset reports, and console logs. It leaves unknown/missing metrics unknown.

## 13. Download Results Back To macOS

Use the file manager to download small artifacts only: `/root/sj-tmp/tapor_runs/EXPERIMENT_SUMMARY.md`, per-run manifests/console logs/dataset reports, and desired `TestDataset_logs` text/pickle files. Download trained checkpoints only when required; do not download raw dataset directories to the Mac. Optionally package results first:

```bash
cd /root/sj-tmp
tar -czf tapor-results-$(date -u +%Y%m%dT%H%M%SZ).tar.gz tapor_runs /workspace/TAPOR/TestDataset_logs /workspace/TAPOR/LogTrainTest
```

Use the browser file manager's Download action on that explicit archive. Confirm its size before transfer.

## 14. Commit Course Project Files

On the Mac project repository, do not commit data, checkpoints, or large logs (the `.gitignore` excludes them):

```bash
git status
git add scripts RUNBOOK.md docs/REPOSITORY_AUDIT.md .gitignore patches
git commit -m "Add reproducible TAPOR baseline workflow"
git push origin HEAD
```

Keep the original TAPOR source and its MIT attribution intact. Record any compatibility patch application in the experiment summary.




PhyPose-Early
Physics attribute maps stacked as multi-channel input to pose encoding & retrieval
Physics fused into pose pipeline early
PhyPose-Trunk
Shared trunk features as physics and pose branches input
Physics fused into pose pipeline at trunk
PhyPose-Late
Parallel physics and pose paths; Late gated residual fusion on joints
Physics fused into pose pipeline late
PhyPose-Map
Build spatial feature map from physics maps; joint queries the map

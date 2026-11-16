#!/usr/bin/env bash
# Exact README TAPOR training settings, excluding only output placement handled by upstream.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-48}"
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || die "EPOCHS must be a positive integer"
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "BATCH_SIZE must be a positive integer"
require_file "$TAPOR_ROOT/traintest.py"
prepare_upstream_artifact_links
prepare_pose_data_links
check_space
"$SCRIPT_DIR/apply_tapor_compat_patch.sh"
RUN_DIR="$(new_run_dir train_full)"
python "$SCRIPT_DIR/inspect_dataset.py" --tapor-root "$TAPOR_ROOT" --data-root "$DATA_ROOT" --report "$RUN_DIR/dataset_inspection.json" --require-complete
python "$SCRIPT_DIR/verify_model.py" --tapor-root "$TAPOR_ROOT" --device cuda --batch-size "$BATCH_SIZE" --report "$RUN_DIR/model_verification.json"

COMMAND="python traintest.py -m tapor -e $EPOCHS -b $BATCH_SIZE -fs 1 -ms 0 -t 0 -lr 0.0001 -hm 0 -fo 1 -s 0 -mt 1 -ls jb -tqdm 1"
if [[ "$EPOCHS" != "200" || "$BATCH_SIZE" != "48" ]]; then
  echo "WARNING: This differs from the official README baseline: epochs=$EPOCHS batch_size=$BATCH_SIZE" | tee "$RUN_DIR/DEVIATION.txt"
fi
echo "For a long run, invoke this script inside tmux. Upstream resume is only weight loading (-wp); it does not restore optimizer, scheduler, epoch, or split state."
start="$(date +%s)"
set +e
(cd "$TAPOR_ROOT" && eval "$COMMAND") 2>&1 | tee "$RUN_DIR/console.log"
code=${PIPESTATUS[0]}
set -e
end="$(date +%s)"
write_manifest "$RUN_DIR/manifest.json" train_full "$COMMAND" "$start" "$end" "$code"
[[ "$code" -eq 0 ]] || exit "$code"
echo "Full baseline command completed. Outputs: $RUN_DIR and $TAPOR_ROOT/{weights,Runs,LogTrainTest}"

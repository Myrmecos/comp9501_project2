#!/usr/bin/env bash
# One epoch on real official training data. This is a pipeline check, not a benchmark run.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

BATCH_SIZE="${BATCH_SIZE:-8}"
require_file "$TAPOR_ROOT/traintest.py"
prepare_upstream_artifact_links
prepare_pose_data_links
check_space
"$SCRIPT_DIR/apply_tapor_compat_patch.sh"
RUN_DIR="$(new_run_dir train_smoke)"
python "$SCRIPT_DIR/inspect_dataset.py" --tapor-root "$TAPOR_ROOT" --data-root "$DATA_ROOT" --report "$RUN_DIR/dataset_inspection.json" --require-complete
python "$SCRIPT_DIR/verify_model.py" --tapor-root "$TAPOR_ROOT" --device cuda --batch-size "$BATCH_SIZE" --report "$RUN_DIR/model_verification.json"

COMMAND="python traintest.py -m tapor -e 1 -b $BATCH_SIZE -fs 1 -ms 0 -t 0 -lr 0.0001 -hm 0 -fo 1 -s 0 -mt 1 -ls jb -tqdm 1"
start="$(date +%s)"
set +e
(cd "$TAPOR_ROOT" && eval "$COMMAND") 2>&1 | tee "$RUN_DIR/console.log"
code=${PIPESTATUS[0]}
set -e
end="$(date +%s)"
write_manifest "$RUN_DIR/manifest.json" train_smoke "$COMMAND" "$start" "$end" "$code"
[[ "$code" -eq 0 ]] || exit "$code"
echo "Smoke run completed. It is NOT a full reproduction. Outputs: $RUN_DIR and $TAPOR_ROOT/{weights,Runs,LogTrainTest}"

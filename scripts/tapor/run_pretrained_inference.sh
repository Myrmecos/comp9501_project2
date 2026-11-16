#!/usr/bin/env bash
# Run the upstream official test protocol; no model code is replaced.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

CHECKPOINT_NAME="${1:-tapor.pth}"
CHECKPOINT="$TAPOR_ROOT/weights/$CHECKPOINT_NAME"
require_file "$TAPOR_ROOT/inference.py"
prepare_upstream_artifact_links
prepare_pose_data_links
require_file "$CHECKPOINT"
check_space

RUN_DIR="$(new_run_dir pretrained_inference)"
REPORT="$RUN_DIR/dataset_inspection.json"
python "$SCRIPT_DIR/inspect_dataset.py" --tapor-root "$TAPOR_ROOT" --data-root "$DATA_ROOT" --report "$REPORT" --require-complete
python "$SCRIPT_DIR/verify_model.py" --tapor-root "$TAPOR_ROOT" --checkpoint "$CHECKPOINT" --device cuda --report "$RUN_DIR/model_verification.json"

COMMAND="python inference.py -m tapor -wp $CHECKPOINT_NAME -fs 1 -ms 0 -hm 0 -fo 1 -mt 1 -v 0"
start="$(date +%s)"
set +e
(cd "$TAPOR_ROOT" && eval "$COMMAND") 2>&1 | tee "$RUN_DIR/console.log"
code=${PIPESTATUS[0]}
set -e
end="$(date +%s)"
write_manifest "$RUN_DIR/manifest.json" pretrained_inference "$COMMAND" "$start" "$end" "$code"
[[ "$code" -eq 0 ]] || exit "$code"
echo "Official inference completed. Console and manifest: $RUN_DIR"
echo "Official result files remain in: $TAPOR_ROOT/TestDataset_logs"

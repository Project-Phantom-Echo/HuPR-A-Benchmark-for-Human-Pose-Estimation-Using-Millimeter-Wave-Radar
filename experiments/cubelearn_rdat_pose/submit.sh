#!/bin/bash
# Submit one learned R-D-A-T run and its dependent best-validation evaluation.
set -euo pipefail
umask 000
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
EPOCHS=${1:-7}
RUN_NAME=${2:-$(id -un)-learned-seed0}
if [[ ! "$EPOCHS" =~ ^[0-9]+$ ]] || (( EPOCHS < 5 || EPOCHS > 10 )); then
    echo 'Choose 5–10 epochs; the measured six-hour recommendation is 7.' >&2
    exit 2
fi
if [[ ! "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo 'Run name must contain only letters, numbers, underscores or hyphens.' >&2
    exit 2
fi
CAMPAIGN=${RDAT_CAMPAIGN:-$ROOT/local/cubelearn-rdat-pose-20261009}
mkdir -p -- "$CAMPAIGN/slurm"
OUT="$CAMPAIGN/$RUN_NAME"
RECORD="$CAMPAIGN/$RUN_NAME-submission"
if [[ -e "$OUT" || -e "$RECORD" ]]; then
    echo "Run or submission already exists: $OUT / $RECORD" >&2
    echo 'Inspect its job IDs before resubmitting; use run.py --resume to continue a checkpoint.' >&2
    exit 1
fi
mkdir -- "$RECORD"
COMMON=(--lpp-lr 0.001 --lr 0.0003 --batch-size 8 --workers 8 --seed 0 --output "$OUT")
TRAIN_CMD=(sbatch --parsable --export=ALL --job-name=cube-rdat-learned
    --time=06:30:00 --output="$CAMPAIGN/slurm/$RUN_NAME-train-%j.out"
    "$SCRIPT_DIR/job.sbatch" --mode train --epochs "$EPOCHS" "${COMMON[@]}")
printf '%q ' "${TRAIN_CMD[@]}" > "$RECORD/train-command.txt"
printf '\n' >> "$RECORD/train-command.txt"
TRAIN=$("${TRAIN_CMD[@]}")
TRAIN=${TRAIN%%;*}
[[ "$TRAIN" =~ ^[0-9]+$ ]]
printf '%s\n' "$TRAIN" > "$RECORD/train-job.txt"
echo "Training job: $TRAIN ($EPOCHS epochs)"
TEST_CMD=(sbatch --parsable --export=ALL --job-name=cube-rdat-test
    --mem=64G --time=00:20:00 --dependency="afterok:$TRAIN" --kill-on-invalid-dep=yes
    --output="$CAMPAIGN/slurm/$RUN_NAME-test-%j.out"
    "$SCRIPT_DIR/job.sbatch" --mode test "${COMMON[@]}")
printf '%q ' "${TEST_CMD[@]}" > "$RECORD/test-command.txt"
printf '\n' >> "$RECORD/test-command.txt"
TEST=$("${TEST_CMD[@]}")
TEST=${TEST%%;*}
[[ "$TEST" =~ ^[0-9]+$ ]]
printf '%s\n' "$TEST" > "$RECORD/test-job.txt"
echo "Test job: $TEST (after successful training, best validation checkpoint)"
echo "Output: $OUT"

#!/bin/bash
#SBATCH --partition=research
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=08:00:00
set -euo pipefail
: "${HUPR_ROOT:?Set HUPR_ROOT to the absolute fork directory}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/hupr-mpl-${USER}"
exec "${HUPR_PYTHON:-$HUPR_ROOT/local/env-hupr/bin/python}" -u "$HUPR_ROOT/run_fast.py" "$@"

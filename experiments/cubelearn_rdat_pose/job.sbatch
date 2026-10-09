#!/bin/bash
#SBATCH --partition=research
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=384G
#SBATCH --time=06:30:00
set -euo pipefail
umask 000
ROOT=/mnt/weka/fgeikyan/rf-perception-papers/hupr
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/hupr-cube-mpl
exec "$ROOT/local/env-hupr/bin/python" -u "$ROOT/experiments/cubelearn_rdat_pose/run.py" "$@"

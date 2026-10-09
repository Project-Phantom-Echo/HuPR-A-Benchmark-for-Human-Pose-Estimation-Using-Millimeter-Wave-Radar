#!/bin/bash
#SBATCH --partition=research
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=384G
#SBATCH --time=00:25:00
#SBATCH --job-name=cube-rdat-benchmark
#SBATCH --output=/mnt/weka/fgeikyan/rf-perception-papers/hupr/local/cubelearn-rdat-pose-20261009/slurm/benchmark-%j.out
set -euo pipefail
umask 000
ROOT=/mnt/weka/fgeikyan/rf-perception-papers/hupr
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/hupr-cube-mpl
PYTHON="$ROOT/local/env-hupr/bin/python"
"$PYTHON" -u "$ROOT/experiments/cubelearn_rdat_pose/check.py"
"$PYTHON" -u "$ROOT/experiments/cubelearn_rdat_pose/run.py" --mode benchmark --lpp-lr 0.001 --benchmark-steps 60 --output "$ROOT/local/cubelearn-rdat-pose-20261009/benchmark-learned"
"$PYTHON" -u "$ROOT/experiments/cubelearn_rdat_pose/run.py" --mode benchmark --lpp-lr 0 --benchmark-steps 60 --output "$ROOT/local/cubelearn-rdat-pose-20261009/benchmark-dft"

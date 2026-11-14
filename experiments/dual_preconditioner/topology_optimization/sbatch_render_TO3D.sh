#!/bin/bash
# Headless render of a 3D TO talk video on one L40S node (VTK EGL), env ~/software/render-venv:
#   sbatch [--export=ALL,TEST=1,OUT=...,TITLE=...] sbatch_render_TO3D.sh <history.nc> [<seg2_history.nc> ...]
#SBATCH --job-name=render_TO3D
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --exclude=g3606,g3608         # GPU ECC errors
#SBATCH --output=/work/classic/fr_ml1145-muffto/logs/render_TO3D-%j.out
set -e
cd ~/software/muFFTTO/experiments/dual_preconditioner/topology_optimization
mkdir -p figures
echo "job $SLURM_JOB_ID host $(hostname) start $(date -Is) TEST=${TEST:-0} OUT=${OUT:-}"
~/software/render-venv/bin/python -u render_TO3D_video.py "$@"
echo "done $(date -Is)"

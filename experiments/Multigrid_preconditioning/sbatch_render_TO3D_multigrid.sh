#!/bin/bash
# Headless render of the multi-grid 3D TO video on one L40S (VTK EGL), env ~/software/render-venv:
#   sbatch [--export=ALL,TEST=1,OUT=...,OBJECTIVES=...] sbatch_render_TO3D_multigrid.sh <history.nc> [<next.nc> ...]
#SBATCH --job-name=FB_render
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --exclude=g3606,g3608,g3610
#SBATCH --output=/work/classic/fr_ml1145-muffto/runs/FABLE/logs/%x-%j.out
set -e
cd ~/software/muFFTTO_fable/experiments/Multigrid_preconditioning
echo "job $SLURM_JOB_ID host $(hostname) start $(date -Is) TEST=${TEST:-0} OUT=${OUT:-}"
~/software/render-venv/bin/python -u render_TO3D_multigrid_video.py "$@"
echo "done $(date -Is)"

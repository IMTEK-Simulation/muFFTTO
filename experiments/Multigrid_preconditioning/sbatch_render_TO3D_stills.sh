#!/bin/bash
# Headless render of the 3D stills of single designs on one L40S (VTK EGL), env ~/software/render-venv:
#   sbatch [--export=ALL,LABELS=...,OUT=...] sbatch_render_TO3D_stills.sh <a.nc> [<b.nc> ...]
#SBATCH --job-name=FB_stills
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --exclude=g3606,g3608,g3610
#SBATCH --output=/work/classic/fr_ml1145-muffto/runs/FABLE/logs/%x-%j.out
set -e
cd ~/software/muFFTTO_fable/experiments/Multigrid_preconditioning
echo "job $SLURM_JOB_ID host $(hostname) start $(date -Is) OUT=${OUT:-}"
~/software/render-venv/bin/python -u render_TO3D_stills.py "$@"
echo "done $(date -Is)"

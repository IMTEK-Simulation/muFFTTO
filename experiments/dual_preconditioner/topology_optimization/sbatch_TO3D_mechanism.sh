#!/bin/bash
# Deformation video of a 3D TO design on one L40S GPU: the 6 load cases (make_TO3D_deformation.py), then the headless
# mechanism render (render_auxetic_mechanism.py, env ~/software/render-venv):
#   sbatch [--export=ALL,LOW_MEMORY=1] sbatch_TO3D_mechanism.sh <design.nc> <out_name> [title]
# -> figures/<out_name>.mp4 + _compressed.png
#SBATCH --job-name=TO3D_mech
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=g3606,g3608
#SBATCH --output=/work/classic/fr_ml1145-muffto/logs/TO3D_mech-%j.out
set -e
DESIGN=$1; OUT=$2; TITLE=${3:-}
cd ~/software/muFFTTO/experiments/dual_preconditioner/topology_optimization
mkdir -p figures
DEFO=/work/classic/fr_ml1145-muffto/runs/TO3D/$OUT.deformation.nc
echo "job $SLURM_JOB_ID host $(hostname) start $(date -Is) design $DESIGN"
(
  source ~/software/muffto_env_gpu.sh
  export DEVICE=gpu CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-$CUDA_ROOT}} XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu MPLBACKEND=Agg
  python -u make_TO3D_deformation.py "$DESIGN" "$DEFO"
)
echo "deformation done $(date -Is)"
env -u PYTHONPATH ~/software/render-venv/bin/python -u render_auxetic_mechanism.py "$DEFO" "figures/$OUT.mp4" "$TITLE"
echo "done $(date -Is)"

#!/bin/bash
# Evaluate the 3D TO objective of a given design once, no optimization (GTOL huge: L-BFGS stops at the start design):
#   sbatch [--export=ALL,STRESS_WEIGHT=..,PF_WEIGHT=..] sbatch_TO3D_evaluate.sh <N> <eta> <design _final.nc> <suffix>
# -> data/example_3D_elasticity_TO/..._eval<suffix>_final.nc (homogenized stresses, norms_pf) + log
#SBATCH --job-name=TO3D_eval
#SBATCH --partition=l40s,rtx,h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --exclude=g3606,g3608
#SBATCH --output=/work/classic/fr_ml1145-muffto/runs/TO3D/logs/eval-%j.out
cd ~/software/muFFTTO/experiments/dual_preconditioner/topology_optimization
source ~/software/muffto_env_gpu.sh
export DEVICE=gpu CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-$CUDA_ROOT}} OMP_NUM_THREADS=1
RUN_SUFFIX=_eval$4 MAXITER=1 NPIX=$1 SOFT_EXP=${SOFT_EXP:-4} ETA=$2 CG_MODE=energy GTOL=1e9 HISTORY_EVERY=1 \
PF_WEIGHT=${PF_WEIGHT:-0.2} INIT_FROM=$3 python -u example_3D_elasticity_TO.py Green_Jacobi

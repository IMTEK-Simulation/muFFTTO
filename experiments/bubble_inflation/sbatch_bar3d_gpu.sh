#!/bin/bash
# One 3D hollow-bar run (pressurized_bar_3D.py, matrix-free Hessian) on one L40S GPU (NEMO2):
#   sbatch sbatch_bar3d_gpu.sh <NNN> <PRECOND>        e.g.  sbatch sbatch_bar3d_gpu.sh 64 G_wall
# Options (environment, via sbatch --export=ALL,...): P_MAX (12), NB_INC (12)
# Output: $MUFFTTO_WS/runs/bar3d/bar3d_n<NNN>_..._<PRECOND>_gpu_<date>/ (run.log, state.npz), job log in $WS/runs/bar3d/logs
#SBATCH --job-name=bar3d_gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --exclude=g3606,g3608
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=/work/classic/fr_ml1145-muffto/runs/bar3d/logs/bar3d_gpu-%j.out
set -e
NNN=$1; P=$2
source ~/software/muffto_env_gpu.sh
export DEVICE=gpu CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-$CUDA_ROOT}} XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
export NNN PRECOND=$P P_MAX=${P_MAX:-12} NB_INC=${NB_INC:-12} MATRIX_FREE=1
export RUNS_DIR=/work/classic/fr_ml1145-muffto/runs/bar3d
cd ~/software/muFFTTO/experiments/bubble_inflation
echo "job $SLURM_JOB_ID on $(hostname): NNN $NNN PRECOND $P P_MAX $P_MAX NB_INC $NB_INC"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python pressurized_bar_3D.py

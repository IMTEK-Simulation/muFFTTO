#!/bin/bash
# Run the TR_SRI finite-strain script on one A100 of bwUniCluster 3.0:
#     sbatch bwuc3_run_gpu.sh
# First do the short check run (see FIRST TEST below) in the development queue.
#SBATCH --job-name=tr_sri_gpu
#SBATCH --partition=gpu_a100_il            # FIRST TEST: dev_gpu_a100_il and --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00                    # queue maximum: 48:00:00
#SBATCH --output=tr_sri_gpu_%j.out
set -eo pipefail

# ---- same modules and venv as in bwuc3_build_gpu_env.sh -------------------------------------------------------------
PY_MODULE=devel/python/3.13.1-gnu-14.2
MPI_MODULE=mpi/openmpi/5.0-gnu-14.2
CUDA_MODULE=CHANGE_ME
VENV=$HOME/venv-muffto-gpu
SCRIPT_DIR=$HOME/muFFTTO/Stefanus/Testing\ file
SCRIPT=TR_SRI_2D_homogenization_finite_strain_elasticity_NeoHookean.py

module purge
module load "$PY_MODULE" "$MPI_MODULE" "$CUDA_MODULE"
source "$VENV/bin/activate"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MPLBACKEND=Agg
cd "$SCRIPT_DIR"
nvidia-smi -L

# FIRST TEST (dev_gpu_a100_il, 30 min): small grid, two increments. Compare with the CPU run of the same command:
#   python -u $SCRIPT --solver cg --device gpu -n 32 -inc 2 --save_per_it
# Production: one GPU, one process (more GPUs need CUDA-aware MPI: not tested)
python -u "$SCRIPT" --solver cg --device gpu -n 128 -inc 40 --save_per_it
#!/bin/bash
# 2D TO (example_2D_elasticity_TO.py) on one NEMO2 GPU, continued from a given design (INIT_FROM): the initial-design
# study (make_TO_CH_starts.py, run_TO_CH_starts.sh) refined in the eta = 4h hierarchy 128^2 -> 256^2 -> 512^2.
#   sbatch [-p l40s] sbatch_TO2D_start_gpu.sh <label> <N> <eta> <init .nc> [run suffix]
# contrast 1e4, w = 0.2 (normalised objective), Green-Jacobi, energy CG stop, up to 2000 L-BFGS iterations.
# Logs: $MUFFTTO_WS/runs/TO_CH_starts/<label>_N<N>.log; data in topology_optimization/data/example_2D_elasticity_TO/
#SBATCH --job-name=TO2D_start
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --time=48:00:00
#SBATCH --exclude=g3606,g3608
#SBATCH --output=/work/classic/fr_ml1145-muffto/runs/TO_CH_starts/slurm-%j.out
LABEL=$1; N=$2; ETA=$3; INIT=$4; SFX=${5:-_start_${LABEL}_from$((N / 2))}
cd ~/software/muFFTTO/experiments/dual_preconditioner/topology_optimization
source ~/software/muffto_env_gpu.sh
export DEVICE=gpu CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-$CUDA_ROOT}} OMP_NUM_THREADS=1
LOG=$MUFFTTO_WS/runs/TO_CH_starts/${LABEL}_N${N}.log
mkdir -p "$(dirname "$LOG")"
echo "job $SLURM_JOB_ID $(hostname) $(date -Is) label $LABEL N $N eta $ETA init $INIT" > "$LOG"
NPIX=$N ETA=$ETA SOFT_EXP=4 PF_NORMALIZED=1 PF_WEIGHT=0.2 CG_MODE=energy GTOL=0 MAXITER=2000 HISTORY_EVERY=10 \
INIT_FROM=$INIT RUN_SUFFIX=$SFX python -u example_2D_elasticity_TO.py Green_Jacobi >> "$LOG" 2>&1
echo "WALL done $(date -Is) exit $?" >> "$LOG"

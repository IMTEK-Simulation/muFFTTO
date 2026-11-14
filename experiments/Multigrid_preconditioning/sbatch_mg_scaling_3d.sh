#!/bin/bash
# 3D multigrid scaling (scaling_case_3d.py) on one GPU (NEMO2): new random 3D Cahn-Hilliard geometry, MG (option (a)
# re-discretised coarse grids) vs Green-Jacobi / Green / dual, library CG with the energy stop (d_min 0).
#   sbatch [-p h200] [--time=..] sbatch_mg_scaling_3d.sh <geometry label> <N> "<eta/h list>"
#   -> $OUT/csv/mg3d_scaling_<geometry>_N<N>.csv (restartable), $OUT/logs/MG3D_<geometry>_N<N>_<jobid>.log
# Env: METHODS (default "mg green_jacobi green dual"), OUT (default $WS/runs/MG_scaling3d)
#SBATCH --job-name=MG3D
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=g3606,g3608         # GPU ECC errors
#SBATCH --output=/work/classic/fr_ml1145-muffto/logs/MG3D-%j.out
G=$1; N=$2; ETAS=$3
WS=/work/classic/fr_ml1145-muffto
OUT=${OUT:-$WS/runs/MG_scaling3d}
METHODS=${METHODS:-"mg green_jacobi green dual"}
source ~/software/muffto_env_gpu.sh
export CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-$CUDA_ROOT}} XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTHONWARNINGS="ignore::RuntimeWarning:muFFTTO.discretization_library"
cd ~/software/muFFTTO/experiments/Multigrid_preconditioning
mkdir -p $OUT/logs
LOG=$OUT/logs/MG3D_${G}_N${N}_${SLURM_JOB_ID}.log
{ echo "job $SLURM_JOB_ID on $(hostname) ($SLURM_JOB_PARTITION), $(date)"; nvidia-smi -L; echo "geometry $G N $N etas $ETAS"; } >> $LOG
for eta in $ETAS; do
  python scaling_case_3d.py --geometry $G --grid $N --eta $eta --methods $METHODS --device gpu --out $OUT >> $LOG 2>&1 \
    || echo "FAILED $G N $N eta $eta (exit $?)" >> $LOG
done
echo "done $(date)" >> $LOG

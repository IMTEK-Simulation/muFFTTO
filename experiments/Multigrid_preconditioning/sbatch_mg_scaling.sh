#!/bin/bash
# Multigrid scaling study on one L40S GPU (NEMO2): phase-field TO design, Green / Green-Jacobi / dual / MG,
# library CG with the energy stop (scaling_case.py).
#   sbatch [--time=HH:MM:SS] sbatch_mg_scaling.sh <N> "<eta/h list>"
#   -> $OUT/csv/mg_scaling_N<N>.csv (one row per eta/method; rows already present are skipped = restart path)
#      $OUT/logs/MG_N<N>_<jobid>.log
# Env (sbatch --export=ALL,...): METHODS (default "mg green dual green_jacobi"), OUT (default $WS/runs/MG_scaling),
#   SMOKE=1 -> --check-N and every eta run twice (the second call must skip everything: restart path)
#   D_MIN (default 0: pure adaptive delay, user decision 2026-11-13)
#SBATCH --job-name=MG_scal
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=g3606,g3608         # GPU ECC errors
#SBATCH --output=/work/classic/fr_ml1145-muffto/logs/MG_scaling-%j.out
N=$1; ETAS=$2
WS=/work/classic/fr_ml1145-muffto
OUT=${OUT:-$WS/runs/MG_scaling}
METHODS=${METHODS:-"mg green dual green_jacobi"}
source ~/software/muffto_env_gpu.sh
export CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-$CUDA_ROOT}} XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
export PYTHONWARNINGS="ignore::RuntimeWarning:muFFTTO.discretization_library"
cd ~/software/muFFTTO/experiments/Multigrid_preconditioning
mkdir -p $OUT/logs
LOG=$OUT/logs/MG_N${N}_${SLURM_JOB_ID}.log
{ echo "job $SLURM_JOB_ID on $(hostname), $(date)"; nvidia-smi -L; echo "N $N etas $ETAS methods $METHODS"; } >> $LOG
for eta in $ETAS; do
  python scaling_case.py --grid $N --eta $eta --methods $METHODS --device gpu --out $OUT --d-min ${D_MIN:-0} ${SMOKE:+--check-N} >> $LOG 2>&1 \
    || echo "FAILED N $N eta $eta (exit $?)" >> $LOG
  if [ -n "${SMOKE:-}" ]; then
    python scaling_case.py --grid $N --eta $eta --methods $METHODS --device gpu --out $OUT --d-min ${D_MIN:-0} >> $LOG 2>&1 \
      || echo "FAILED restart N $N eta $eta" >> $LOG
  fi
done
echo "done $(date)" >> $LOG

#!/bin/bash
# Fracture-Bench (Krishnan et al. 2026) periodic analogues, mesh refinement at fixed l, one L40S GPU (NEMO2):
#   sbatch sbatch_fb_gpu.sh 1 <N>  <P>   Dataset 1, fb_dataset1_AT2.py, N^2 grid (l/h = 0.87 N / 128), P = Green|dual
#   sbatch sbatch_fb_gpu.sh 2 <NX> <P>   Dataset 2, fb_dataset2_pfczm.py, NX x 2NX grid (b/h = 1.28 NX / 128)
# Load increments as theirs (D1 biaxial 1e-4, D2 1.25e-5); D1 LOAD_STEPS default 120 (theirs 44: no crack in our
# periodic analogue), stops when cut through; D2 LOAD_STEPS default 80.
# Options (env, sbatch --export=ALL,...): LOAD_STEPS, RUN_SUFFIX (smoke test), TIME_BUDGET (D2 checkpoint, s)
# Outputs: fracture_bench/exp_data, figures -> $RUNS/{exp_data,figures}; logs $RUNS/logs/FB<ds>_N<N>_<P><suffix>.log
#SBATCH --job-name=fb_gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --exclude=g3606,g3608         # GPU ECC errors
#SBATCH --output=/work/classic/fr_ml1145-muffto/runs/fracture_bench/logs/slurm_fb-%j.out
set -e
DS=$1; N=$2; P=$3
RUNS=/work/classic/fr_ml1145-muffto/runs/fracture_bench
source ~/software/muffto_env_gpu.sh
export DEVICE=gpu CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-$CUDA_ROOT}} XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
cd ~/software/muFFTTO/experiments/dual_preconditioner/fracture_bench
echo "job $SLURM_JOB_ID host $(hostname) start $(date -Is) dataset $DS N=$N $P suffix=${RUN_SUFFIX:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "$DS" = 1 ]; then
  STEPS=${LOAD_STEPS:-120}
  export NPIX=$N MICROSTRUCTURE=fb_hole HOLES=1 HOLE_STIFFNESS=1e-3 K_RESIDUAL=1e-3 LAME=0,1 GC=7.8227e-6 \
         LENGTH_SCALE=0.0068182 LOAD_DIR=biax LOAD_STEPS=$STEPS STOP_FRACTION=0.2 \
         LOAD_MAX=$(python3 -c "print(1e-4 * $STEPS)")
  LOG=$RUNS/logs/FB1_N${N}_${P}${RUN_SUFFIX:-}.log
  python fb_dataset1_AT2.py $P > $LOG 2>&1
else
  export GRID=$N,$((2 * N)) LOAD_STEPS=${LOAD_STEPS:-80} TIME_BUDGET=${TIME_BUDGET:-84000}
  LOG=$RUNS/logs/FB2_N${N}x$((2 * N))_${P}${RUN_SUFFIX:-}.log
  python fb_dataset2_pfczm.py $P > $LOG 2>&1
fi
echo "end $(date -Is) rc=$?"

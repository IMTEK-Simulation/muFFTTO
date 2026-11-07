#!/bin/bash
# One core-shell battery run (Tu et al. 2026 length scales, run_battery_tu.sh settings) on one GPU (NEMO2):
#   sbatch [-p h200 --mem=200G] sbatch_battery_gpu.sh <DIM 2|3> <N> <Green|Green_Jacobi|dual>
#     -> logs/battery_tu_N<N>_<P>[_3D][_r<SEG>].log, exp_data/battery_N<N>_<P>_ms0.01[_3D]_tu_{steps,iterations}.csv
# Settings: 6 um cell, R_core = 2 um, shell 0.2 R, l = 0.23 / 0.27 um (fixed for every grid), l_zeta = 0.1 um, a = 0.3 h,
# 1C lithiation (pixel-wise CC -> CV), dt = 22 s, until 4969 s, MAX_STAGGERED 500; diffusion shadow solves (all three
# diffusion preconditioners) in the dual runs only.
# Options (environment, via sbatch --export=ALL,...): T_MAX (smoke test), EXTRA_SUFFIX (smoke test: _smoke),
#   SNAP_EVERY (default 10).
# 48 h limit: the run is stopped after BUDGET s (default 47 h 30 min) and the job resubmits itself (SEG+1, RESTART=1)
# from <output>_checkpoint.npz (written every CHECKPOINT_SECONDS = 1800 s of wall time); at most MAXSEG (3) segments.
# exp_data and logs -> symlinks into /work/classic/fr_ml1145-muffto/runs/battery/
#SBATCH --job-name=battery_gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --exclude=g3606,g3608         # GPU ECC errors
#SBATCH --output=/work/classic/fr_ml1145-muffto/logs/battery_gpu-%j.out
set -e
D=$1; N=$2; P=$3
SEG=${SEG:-0}; BUDGET=${BUDGET:-171000}; MAXSEG=${MAXSEG:-3}
source ~/software/muffto_env_gpu.sh
export DEVICE=gpu CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-$CUDA_ROOT}} XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
cd ~/software/muFFTTO/experiments/dual_preconditioner/battery_particle_fracture
SHADOW=0; [ "$P" = dual ] && SHADOW=1
TAG=""; [ "$D" = 3 ] && TAG=_3D
export DIM=$D NPIX=$N L_CELL_UM=6 R_CORE_UM=2 H_BAR=0.2 A_BAR=0.3 L_CORE_UM=0.23 L_SHELL_UM=0.27 L_ZETA_UM=0.1 \
       DT=0.02 T_MAX=${T_MAX:-4.5} MATRIX_STIFFNESS=1e-2 MAX_STAGGERED=500 DIFF_SHADOW=$SHADOW \
       SNAP_EVERY=${SNAP_EVERY:-10} CHECKPOINT_SECONDS=${CHECKPOINT_SECONDS:-1800} RUN_SUFFIX=_tu${EXTRA_SUFFIX:-}
if [ "$SEG" -gt 0 ]; then export RESTART=1; fi
SEGTAG=""; if [ "$SEG" -gt 0 ]; then SEGTAG=_r$SEG; fi
LOG=logs/battery_tu_N${N}_${P}${TAG}${EXTRA_SUFFIX:-}${SEGTAG}.log
STEM=exp_data/battery_N${N}_${P}_ms0.01${TAG}_tu${EXTRA_SUFFIX:-}
echo "job $SLURM_JOB_ID host $(hostname) start $(date -Is) DIM=$D N=$N $P SEG=$SEG log=$PWD/$LOG"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
( sleep 120; while true; do echo "$(date +%H:%M) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ') util $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)"; sleep 600; done ) &
mon=$!
start=$(date +%s)
rc=0
timeout -s TERM -k 120 "$BUDGET" python -u example_2D_battery_particle_fracture_AT2.py "$P" > "$LOG" 2>&1 || rc=$?
kill $mon 2>/dev/null || true
echo "done $(date -Is) exit $rc wall $(( $(date +%s) - start )) s"
ls -la ${STEM}* 2>/dev/null || true
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  if [ "$SEG" -ge $((MAXSEG - 1)) ]; then echo "time limit, MAXSEG=$MAXSEG reached: not continuing"; exit 1; fi
  if [ ! -f "${STEM}_checkpoint.npz" ]; then echo "time limit, no checkpoint: not continuing"; exit 1; fi
  NEXT=$(sbatch --parsable -p "$SLURM_JOB_PARTITION" --mem="${SLURM_MEM_PER_NODE}M" --job-name="${SLURM_JOB_NAME}_r$((SEG + 1))" \
         --export=ALL,SEG=$((SEG + 1)) sbatch_battery_gpu.sh "$D" "$N" "$P")
  echo "time limit: continued as job $NEXT (segment $((SEG + 1)))"
  echo "$NEXT ${SLURM_JOB_NAME}_r$((SEG + 1)) continuation of $SLURM_JOB_ID" >> /work/classic/fr_ml1145-muffto/runs/battery/jobs.txt
  exit 0
fi
exit $rc

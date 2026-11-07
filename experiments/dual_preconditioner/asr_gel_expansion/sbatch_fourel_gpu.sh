#!/bin/bash
# One Fourel-model run (fourel_nonlocal_damage.py) on one GPU of NEMO2, in the self-contained run folder
# /work/classic/fr_ml1145-muffto/runs/asr_fourel (scripts + geometry copied there by push_nemo_fourel.sh; library =
# ~/software/muFFTTO via MUFFTTO_ROOT):
#   sbatch [-p h200 --mem=200G] sbatch_fourel_gpu.sh <DIM> <N> <MS|Green|Green_Jacobi|dual>
# 2D -> l40s (default partition), 3D 256^3 -> h200 --mem=200G.
# Options (sbatch --export=ALL,...): LOADING (free), TOL_MS (1e-4), BATCH (0), K_RESIDUAL (1e-4), LOAD_STEPS (200),
#   EPS_MAX (0.05), STEPS_RUN (all), CHECK_EVERY (0), SNAP_EVERY (2D 1, 3D 1 mid-plane), SNAP_FULL (3D full field, 10),
#   GEOMETRY_NPZ (geometry/fourel<DIM>D_s1.npz), EXTRA_SUFFIX (appended to the run name and the log)
# 48 h limit: stopped after BUDGET s (default 47 h 30 min); then resubmits itself (SEG+1) with
# RESTART_FROM = <run>_checkpoint.npz (written every CHECKPOINT_SECONDS at a load-step end), at most MAXSEG (3) segments.
#SBATCH --job-name=FOUREL_gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --exclude=g3606,g3608         # GPU ECC errors
#SBATCH --output=/work/classic/fr_ml1145-muffto/runs/asr_fourel/slurm/FOUREL_gpu-%j.out
set -e
D=$1; N=$2; P=$3
SEG=${SEG:-0}; BUDGET=${BUDGET:-171000}; MAXSEG=${MAXSEG:-3}
BASE_EXTRA=${EXTRA_SUFFIX:-}
R=/work/classic/fr_ml1145-muffto/runs/asr_fourel
source ~/software/muffto_env_gpu.sh
export DEVICE=gpu CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-$CUDA_ROOT}} XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
export MUFFTTO_ROOT=$HOME/software/muFFTTO OUT_DIR=$R/exp_data
export DIM=$D NPIX=$N RUN_SUFFIX=$BASE_EXTRA CHECKPOINT_SECONDS=${CHECKPOINT_SECONDS:-3600}
cd $R
mkdir -p exp_data logs slurm
# run name exactly as fourel_nonlocal_damage.py builds it (for the log and the checkpoint)
STEM=$(python -c "
import os
k = float(os.environ.get('K_RESIDUAL', 1e-4)); t = float(os.environ.get('TOL_MS', 1e-4))
print(f'FOUREL{\"3D\" if $D == 3 else \"\"}_N$N'+f'_$P'+f'_{os.environ.get(\"LOADING\", \"free\")}_k{k:g}'
      + ('_batch' if int(os.environ.get('BATCH', 0)) else '') + (f'_tolms{t:g}' if '$P' == 'MS' and t != 1e-4 else '')
      + os.environ.get('RUN_SUFFIX', ''))")
LOG=logs/${STEM}.log
[ "$SEG" -gt 0 ] && LOG=logs/${STEM}_r$SEG.log || true
echo "job $SLURM_JOB_ID host $(hostname) start $(date -Is) DIM=$D N=$N $P stem=$STEM log=$R/$LOG RESTART_FROM=${RESTART_FROM:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
( sleep 120; while true; do echo "$(date +%H:%M) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ') util $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)"; sleep 600; done ) &
mon=$!
start=$(date +%s)
rc=0
timeout -s TERM -k 120 "$BUDGET" python -u fourel_nonlocal_damage.py "$P" > "$LOG" 2>&1 || rc=$?
kill $mon 2>/dev/null || true
echo "done $(date -Is) exit $rc wall $(( $(date +%s) - start )) s"
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  if [ "$SEG" -ge $((MAXSEG - 1)) ]; then echo "time limit, MAXSEG=$MAXSEG reached: not continuing"; exit 1; fi
  FROM=$R/exp_data/${STEM}_checkpoint.npz
  if [ ! -f "$FROM" ]; then echo "time limit, no checkpoint: not continuing"; exit 1; fi
  NEXT=$(sbatch --parsable -p "$SLURM_JOB_PARTITION" --mem="${SLURM_MEM_PER_NODE}M" --job-name="${SLURM_JOB_NAME}_r$((SEG + 1))" \
         --export=ALL,SEG=$((SEG + 1)),RESTART_FROM=$FROM,EXTRA_SUFFIX=$BASE_EXTRA $R/sbatch_fourel_gpu.sh "$D" "$N" "$P")
  echo "time limit: continued as job $NEXT (segment $((SEG + 1)), RESTART_FROM=$FROM)"
  echo "$NEXT ${D}D N=$N $P segment $((SEG + 1)) (continues $SLURM_JOB_ID)" >> $R/jobs.txt
  exit 0
fi
exit $rc

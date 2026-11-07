#!/bin/bash
# One phase-field fatigue run (example_2D_phase_field_fatigue_AT2.py) on one GPU (NEMO2): the mesh-size study with
# Carrara et al.'s real parameters (README.md, "NEMO mesh study"), fixed l = L/64:
#   sbatch [-p h200 --mem 200G] sbatch_fatigue_gpu.sh 2D|3D <N> <precond>   -> logs/FAT_nemo<D>_N<N>_<P>.log
#   2D: plane stress, glass-bead disks (geometry/beads2D_2048.npy), l/h = N / 64 (32^2 ... 1024^2: 0.5 ... 16)
#   3D: 3D Lame, glass-bead spheres (geometry/beads3D_256.npy), through pre-crack, l/h = N / 64 (32^3 ... 256^3)
# Options (environment, via sbatch --export=ALL,...):
#   CYCLES=<n>                override (smoke test), EXTRA_SUFFIX=<s> appended to RUN_SUFFIX and the log name
#   AMP=<e>                   strain amplitude (default 2D 2.0e-3 = 0.25 x the monotonic peak 8e-3; 3D 2.2e-3 = 0.25 x 9e-3)
#   RESTART_FROM=<.npz/.nc>   continue a run, with SEG=<k> (output and log get the suffix _r<k>)
# 48 h limit: the run is stopped after BUDGET s (default 47 h 30 min) and the job resubmits itself (SEG+1) with
# RESTART_FROM = this segment's <output>_checkpoint.npz, else its .nc; at most MAXSEG (default 3) segments
# Output: exp_data, figures, logs of this folder -> symlinks into /work/classic/fr_ml1145-muffto/runs/fatigue/
#SBATCH --job-name=FAT_gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --exclude=g3606,g3608         # GPU ECC errors
#SBATCH --output=/work/classic/fr_ml1145-muffto/logs/FAT_gpu-%j.out
set -e
D=$1; N=$2; P=$3
SEG=${SEG:-0}; BUDGET=${BUDGET:-171000}; MAXSEG=${MAXSEG:-3}
BASE_EXTRA=${EXTRA_SUFFIX:-}
EXTRA_SUFFIX=$BASE_EXTRA; [ "$SEG" -gt 0 ] && EXTRA_SUFFIX=${EXTRA_SUFFIX}_r$SEG || true
source ~/software/muffto_env_gpu.sh
export DEVICE=gpu CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-$CUDA_ROOT}} XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
cd ~/software/muFFTTO/experiments/dual_preconditioner/phase_field_fatigue
# Carrara et al. 2020, Sec. 4.3: E = 6 GPa, nu = 0.22, Gc = 2.28 N/mm, l = 0.2 mm, alpha_T = Gc / (12 l); cell L = 64 l
# = 12.8 mm (units L, E); glass beads E = 70 GPa, nu = 0.22 (stiffness 11.6667, tough), r = 5 - 8 l, 20 %
export INCLUSION_STIFFNESS=11.6667 NOTCH=0.375,0.625,0.5 NOTCH_MODE=history LENGTH_SCALE=0.015625 GC=2.96875e-5 \
       LOAD_DIR=yy K_RESIDUAL=1e-4 FATIGUE=asymptotic CYCLES=${CYCLES:-1500} MAX_STAGGERED=500 STOP_FRACTION=0.3 \
       OUTPUT_FIELDS=damage NC_EVERY=${NC_EVERY:-5} CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-50} NPIX=$N
case $D in
  2D) export DIM=2 GEOMETRY_FILE=geometry/beads2D_2048.npy LAME=0.231190,0.409836 AMPLITUDE=${AMP:-2.0e-3} ;;
  3D) export DIM=3 GEOMETRY_FILE=geometry/beads3D_256.npy LAME=0.322011,0.409836 AMPLITUDE=${AMP:-2.2e-3} ;;
  *) echo "first argument must be 2D or 3D"; exit 2 ;;
esac
export RUN_SUFFIX=_nemo${D}${EXTRA_SUFFIX}
LOG=logs/FAT_nemo${D}_N${N}_${P}${EXTRA_SUFFIX}.log
echo "job $SLURM_JOB_ID host $(hostname) start $(date -Is) $D N=$N $P log=$PWD/$LOG RESTART_FROM=${RESTART_FROM:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
( sleep 120; while true; do echo "$(date +%H:%M) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ') util $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)"; sleep 600; done ) &
mon=$!
start=$(date +%s)
rc=0
timeout -s TERM -k 120 "$BUDGET" python -u example_2D_phase_field_fatigue_AT2.py "$P" > "$LOG" 2>&1 || rc=$?
kill $mon 2>/dev/null || true
echo "done $(date -Is) exit $rc wall $(( $(date +%s) - start )) s"
STEM=$(ls -t exp_data/FAT_N${N}_${P}_*_nemo${D}${EXTRA_SUFFIX}_precrack_*_cycles.csv 2>/dev/null | head -1 | sed 's/_cycles.csv$//')
ls -la ${STEM}* 2>/dev/null || true
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  if [ "$SEG" -ge $((MAXSEG - 1)) ]; then echo "time limit, MAXSEG=$MAXSEG reached: not continuing"; exit 1; fi
  if [ -f "${STEM}_checkpoint.npz" ]; then FROM=$PWD/${STEM}_checkpoint.npz
  elif [ -f "${STEM}.nc" ]; then FROM=$PWD/${STEM}.nc
  else echo "time limit, no checkpoint or .nc: not continuing"; exit 1; fi
  NEXT=$(sbatch --parsable -p "$SLURM_JOB_PARTITION" --mem="${SLURM_MEM_PER_NODE}M" --job-name="${SLURM_JOB_NAME}_r$((SEG + 1))" \
         --export=ALL,SEG=$((SEG + 1)),RESTART_FROM=$FROM,EXTRA_SUFFIX=$BASE_EXTRA sbatch_fatigue_gpu.sh "$D" "$N" "$P")
  echo "time limit: continued as job $NEXT (segment $((SEG + 1)), RESTART_FROM=$FROM)"
  echo "$NEXT $D N=$N $P segment $((SEG + 1)) (continuation of $SLURM_JOB_ID)" >> /work/classic/fr_ml1145-muffto/runs/fatigue/jobs.txt
  exit 0
fi
exit $rc

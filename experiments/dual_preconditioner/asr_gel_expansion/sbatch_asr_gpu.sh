#!/bin/bash
# One ASR mesh-study run (asr_AT2_mesh_study.py) on one GPU of NEMO2:
#   sbatch [-p h200 --mem=200G] sbatch_asr_gpu.sh <DIM> <N> <precond>
#   -> logs/ASRmesh_<DIM>D_N<N>_<P>_l0.015625_cheap<EXTRA>.log, exp_data/ASRmesh_<DIM>D_N<N>_<P>_l0.015625_cheap*
# Settings: l = 1.5625 mm fixed (LENGTH_SCALE=0.015625), cheap zero-stress control, dV/V 0 -> 0.064 in 64 steps,
# geometry/falsafi_s1.npz (2D) / geometry/asr3D_s1.npz (3D), MAX_STAGGERED 500.
# Options (sbatch --export=ALL,...): LOAD_STEPS / DVV_MAX (smoke test), EXTRA_SUFFIX (appended to RUN_SUFFIX and log)
# 48 h limit: stopped after BUDGET s (default 47 h 30 min); then resubmits itself (SEG+1) with
# RESTART_FROM = <run>_checkpoint.npz (written every CHECKPOINT_SECONDS at a load-step end), at most MAXSEG (3) segments.
# exp_data/ and logs/ of the experiment folder are symlinks into /work/classic/fr_ml1145-muffto/runs/asr/
#SBATCH --job-name=ASR_gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --exclude=g3606,g3608         # GPU ECC errors
#SBATCH --output=/work/classic/fr_ml1145-muffto/runs/asr/slurm/ASR_gpu-%j.out
set -e
D=$1; N=$2; P=$3
SEG=${SEG:-0}; BUDGET=${BUDGET:-171000}; MAXSEG=${MAXSEG:-3}
BASE_EXTRA=${EXTRA_SUFFIX:-}
source ~/software/muffto_env_gpu.sh
export DEVICE=gpu CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-$CUDA_ROOT}} XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
cd ~/software/muFFTTO/experiments/dual_preconditioner/asr_gel_expansion
export DIM=$D NPIX=$N LENGTH_SCALE=0.015625 CONTROL=cheap MAX_STAGGERED=500 \
       LOAD_STEPS=${LOAD_STEPS:-64} DVV_MAX=${DVV_MAX:-0.064} RUN_SUFFIX=$BASE_EXTRA \
       CHECKPOINT_SECONDS=${CHECKPOINT_SECONDS:-3600}
STEM=ASRmesh_${D}D_N${N}_${P}_l0.015625_cheap${BASE_EXTRA}
LOG=logs/${STEM}.log
[ "$SEG" -gt 0 ] && LOG=logs/${STEM}_r$SEG.log || true
echo "job $SLURM_JOB_ID host $(hostname) start $(date -Is) DIM=$D N=$N $P log=$PWD/$LOG RESTART_FROM=${RESTART_FROM:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
( sleep 120; while true; do echo "$(date +%H:%M) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ') util $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)"; sleep 600; done ) &
mon=$!
start=$(date +%s)
rc=0
timeout -s TERM -k 120 "$BUDGET" python -u asr_AT2_mesh_study.py "$P" > "$LOG" 2>&1 || rc=$?
kill $mon 2>/dev/null || true
echo "done $(date -Is) exit $rc wall $(( $(date +%s) - start )) s"
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  if [ "$SEG" -ge $((MAXSEG - 1)) ]; then echo "time limit, MAXSEG=$MAXSEG reached: not continuing"; exit 1; fi
  FROM=$PWD/exp_data/${STEM}_checkpoint.npz
  if [ ! -f "$FROM" ]; then echo "time limit, no checkpoint: not continuing"; exit 1; fi
  NEXT=$(sbatch --parsable -p "$SLURM_JOB_PARTITION" --mem="${SLURM_MEM_PER_NODE}M" --job-name="${SLURM_JOB_NAME}_r$((SEG + 1))" \
         --export=ALL,SEG=$((SEG + 1)),RESTART_FROM=$FROM,EXTRA_SUFFIX=$BASE_EXTRA sbatch_asr_gpu.sh "$D" "$N" "$P")
  echo "time limit: continued as job $NEXT (segment $((SEG + 1)), RESTART_FROM=$FROM)"
  echo "$NEXT ${D}D N=$N $P segment $((SEG + 1)) (continues $SLURM_JOB_ID)" >> /work/classic/fr_ml1145-muffto/runs/asr/jobs.txt
  exit 0
fi
exit $rc

#!/bin/bash
# One asr_pf_fourel.py run on one GPU of NEMO2:
#   sbatch [-p h200 --mem=200G] sbatch_asrpf_gpu.sh <DIM> <N> <VARIANT> <precond>
#   VARIANT: fourel (A, AT1 Fourel-matched) | czm (B, PF-CZM real Gf, volumetric driving) | czmR (B, Rankine driving)
#   -> $R/logs/ASRpf_<VARIANT>[_3D]_N<N>_<P><EXTRA>.log, $R/exp_data/ASRpf_<VARIANT>[_3D]_N<N>_<P><EXTRA>_*
#      with R = /work/classic/fr_ml1145-muffto/runs/asr_pf_fourel
# Defaults: eps_ASR 0 -> EPS_MAX (A 0.02, B 0.03) in steps of 0.025 % (A) / 0.05 % (B), MAX_STAGGERED 500,
# k_res 1e-4, geometry/fourel<DIM>D_s1.npz, mean stress -1 MPa, checkpoint every CHECKPOINT_SECONDS (3600) at a
# load-step end. Options (sbatch --export=ALL,...): EPS_MAX, LOAD_STEPS (smoke test), EXTRA_SUFFIX, K_RESIDUAL.
# 48 h limit: stopped after BUDGET s (default 47 h 30 min); then resubmits itself (SEG+1) with
# RESTART_FROM = <run>_checkpoint.npz, at most MAXSEG (3) segments.
#SBATCH --job-name=ASRpf_gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --exclude=g3606,g3608         # GPU ECC errors
#SBATCH --output=/work/classic/fr_ml1145-muffto/runs/asr_pf_fourel/slurm/ASRpf_gpu-%j.out
set -e
D=$1; N=$2; V=$3; P=$4
SEG=${SEG:-0}; BUDGET=${BUDGET:-171000}; MAXSEG=${MAXSEG:-3}
BASE_EXTRA=${EXTRA_SUFFIX:-}
R=/work/classic/fr_ml1145-muffto/runs/asr_pf_fourel
mkdir -p $R/exp_data $R/logs $R/slurm
source ~/software/muffto_env_gpu.sh
export DEVICE=gpu CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-$CUDA_ROOT}} XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
cd ~/software/muFFTTO/experiments/dual_preconditioner/asr_gel_expansion
case $V in
  fourel) DEF_EPS=0.02; DEF_STEPS=80 ;;
  *)      DEF_EPS=0.03; DEF_STEPS=60 ;;
esac
export DIM=$D NPIX=$N VARIANT=$V MAX_STAGGERED=500 OUT_DIR=$R/exp_data \
       EPS_MAX=${EPS_MAX:-$DEF_EPS} LOAD_STEPS=${LOAD_STEPS:-$DEF_STEPS} RUN_SUFFIX=$BASE_EXTRA \
       CHECKPOINT_SECONDS=${CHECKPOINT_SECONDS:-3600} SNAP_FULL=${SNAP_FULL:-10}
[ "$D" = 3 ] && STEM=ASRpf_${V}_3D_N${N}_${P}${BASE_EXTRA} || STEM=ASRpf_${V}_N${N}_${P}${BASE_EXTRA}
LOG=$R/logs/${STEM}.log
[ "$SEG" -gt 0 ] && LOG=$R/logs/${STEM}_r$SEG.log || true
echo "job $SLURM_JOB_ID host $(hostname) start $(date -Is) DIM=$D N=$N $V $P log=$LOG RESTART_FROM=${RESTART_FROM:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
( sleep 120; while true; do echo "$(date +%H:%M) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ') util $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)"; sleep 600; done ) &
mon=$!
start=$(date +%s)
rc=0
timeout -s TERM -k 120 "$BUDGET" python -u asr_pf_fourel.py "$P" > "$LOG" 2>&1 || rc=$?
kill $mon 2>/dev/null || true
echo "done $(date -Is) exit $rc wall $(( $(date +%s) - start )) s"
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  if [ "$SEG" -ge $((MAXSEG - 1)) ]; then echo "time limit, MAXSEG=$MAXSEG reached: not continuing"; exit 1; fi
  FROM=$R/exp_data/${STEM}_checkpoint.npz
  if [ ! -f "$FROM" ]; then echo "time limit, no checkpoint: not continuing"; exit 1; fi
  NEXT=$(sbatch --parsable -p "$SLURM_JOB_PARTITION" --mem="${SLURM_MEM_PER_NODE}M" --job-name="${SLURM_JOB_NAME}_r$((SEG + 1))" \
         --export=ALL,SEG=$((SEG + 1)),RESTART_FROM=$FROM,EXTRA_SUFFIX=$BASE_EXTRA sbatch_asrpf_gpu.sh "$D" "$N" "$V" "$P")
  echo "time limit: continued as job $NEXT (segment $((SEG + 1)), RESTART_FROM=$FROM)"
  echo "$NEXT ${D}D N=$N $V $P segment $((SEG + 1)) (continues $SLURM_JOB_ID)" >> $R/jobs.txt
  exit 0
fi
exit $rc

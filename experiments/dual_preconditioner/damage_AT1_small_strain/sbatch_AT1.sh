#!/bin/bash
# One AT1 run (B2 nine polydisperse pores) on NEMO2, CPU or GPU -- the partition decides:
#   sbatch -p cpu                    [-n <ranks>] sbatch_AT1.sh <N> <LH> <precond>
#   sbatch -p l40s,h200,rtx,mi300a  --gres=gpu:1 sbatch_AT1.sh <N> <LH> <precond>   (starts on the first free GPU)
# <LH> = l/h (l = LH / N, e.g. 2, 0.5, 0.25) or L32 (fixed l = 1/32, the mesh-convergence series)
# Only iterations and results are compared in this study (no timings), so any partition / node type is fine.
# Output: exp_data, logs -> symlinks into $MUFFTTO_WS/runs/AT1/; log name = output name.
# Options (environment): LOAD_MAX (0.32), LOAD_STEPS (320) -- load increment 1e-3, the stop rule ends at cut-through;
#   EXTRA_SUFFIX=<s> appended to RUN_SUFFIX and the log name (smoke tests).
# 24 h segments: on BUDGET (default 23 h 30 min) the job resubmits itself (SEG+1, RESTART_FROM = the checkpoint),
# at most MAXSEG (default 3) segments. Restart is serial only -> CPU jobs with > 1 rank do not continue.
#SBATCH --job-name=AT1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --exclude=g3606,g3608         # GPU ECC errors
#SBATCH --output=/work/classic/fr_ml1145-muffto/logs/AT1-%j.out
set -e
N=$1; LH=$2; P=$3
SEG=${SEG:-0}; BUDGET=${BUDGET:-84600}; MAXSEG=${MAXSEG:-3}
BASE_EXTRA=${EXTRA_SUFFIX:-}
EXTRA_SUFFIX=$BASE_EXTRA; [ "$SEG" -gt 0 ] && EXTRA_SUFFIX=${EXTRA_SUFFIX}_r$SEG || true
case $SLURM_JOB_PARTITION in
  l40s|h200|rtx|mi300a)
    source ~/software/muffto_env_gpu.sh
    export DEVICE=gpu CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-$CUDA_ROOT}} XLA_PYTHON_CLIENT_PREALLOCATE=false
    RUN="python -u" ;;
  *)
    source ~/software/muffto_env.sh
    export DEVICE=cpu
    RUN=$([ "${SLURM_NTASKS:-1}" -gt 1 ] && echo "srun -n $SLURM_NTASKS python -u" || echo "python -u") ;;
esac
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
cd ~/software/muFFTTO/experiments/dual_preconditioner/damage_AT1_small_strain
if [ "$LH" = L32 ]; then LS=0.03125; else LS=$(awk -v a="$LH" -v n="$N" 'BEGIN{printf "%g", a / n}'); fi
export MODEL=AT1 MICROSTRUCTURE=polydisperse_pores NPIX=$N LENGTH_SCALE=$LS \
       LOAD_MAX=${LOAD_MAX:-0.32} LOAD_STEPS=${LOAD_STEPS:-320} RUN_SUFFIX=${EXTRA_SUFFIX}
LOG=logs/AT1_N${N}_${P}_polydisperse_pores_l${LS}${EXTRA_SUFFIX}.log
echo "job $SLURM_JOB_ID partition $SLURM_JOB_PARTITION host $(hostname) start $(date -Is) N=$N l/h=$LH l=$LS $P" \
     "ranks ${SLURM_NTASKS:-1} log=$PWD/$LOG RESTART_FROM=${RESTART_FROM:-}"
[ "$DEVICE" = gpu ] && { nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || rocm-smi --showproductname 2>/dev/null | grep -m1 GPU || true; }
start=$(date +%s); rc=0
timeout -s TERM -k 120 "$BUDGET" $RUN example_2D_phase_field_fracture_AT1.py "$P" > "$LOG" 2>&1 || rc=$?
echo "done $(date -Is) exit $rc wall $(( $(date +%s) - start )) s"
STEM=exp_data/$(basename "$LOG" .log)
ls -la ${STEM}* 2>/dev/null || true
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  if [ "${SLURM_NTASKS:-1}" -gt 1 ]; then echo "time limit, MPI run: restart is serial only, not continuing"; exit 1; fi
  if [ "$SEG" -ge $((MAXSEG - 1)) ]; then echo "time limit, MAXSEG=$MAXSEG reached: not continuing"; exit 1; fi
  if [ -f "${STEM}_checkpoint.npz" ]; then FROM=$PWD/${STEM}_checkpoint.npz
  elif [ -f "${STEM}.nc" ]; then FROM=$PWD/${STEM}.nc
  else echo "time limit, no checkpoint or .nc: not continuing"; exit 1; fi
  GRES=""; if [ "$DEVICE" = gpu ]; then GRES="--gres=gpu:1"; fi
  NEXT=$(sbatch --parsable -p "$SLURM_JOB_PARTITION" $GRES --mem="${SLURM_MEM_PER_NODE}M" \
         --job-name="${SLURM_JOB_NAME}_r$((SEG + 1))" \
         --export=ALL,SEG=$((SEG + 1)),RESTART_FROM=$FROM,EXTRA_SUFFIX=$BASE_EXTRA sbatch_AT1.sh "$N" "$LH" "$P")
  echo "time limit: continued as job $NEXT (segment $((SEG + 1)), RESTART_FROM=$FROM)"
  exit 0
fi
exit $rc

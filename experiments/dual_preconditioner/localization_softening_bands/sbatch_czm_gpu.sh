#!/bin/bash
# PF-CZM concrete mesh-size study on one GPU (NEMO2), pfczm_concrete.py.
#   sbatch [-p h200 --mem 200G] sbatch_czm_gpu.sh <spec> [<spec> ...]
#   spec = DIM:N:PRECOND:LOAD_DIR   e.g. 2:256:dual:xx   (several specs: run one after the other, small grids)
# Options (environment, sbatch --export=ALL,...): K_RESIDUAL (1e-5), LOAD_STEPS, LOAD_MAX, RUN_SUFFIX (smoke test).
# A single-spec job that hits the time budget (TIME_BUDGET = BUDGET s, default 47 h) writes a checkpoint, exits with
# code 3 and is resubmitted with RESTART_FROM=<checkpoint> (SEG+1, at most MAXSEG = 3 segments). Logs of segment k>0
# get the suffix _r<k>; the CSVs are appended across segments.
# Outputs: exp_data -> /work/classic/fr_ml1145-muffto/runs/czm/exp_data, logs in /work/classic/fr_ml1145-muffto/runs/czm/logs
#SBATCH --job-name=czm_gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --exclude=g3606,g3608         # GPU ECC errors
#SBATCH --output=/work/classic/fr_ml1145-muffto/runs/czm/logs/slurm_czm-%j.out
set -e
SEG=${SEG:-0}; BUDGET=${BUDGET:-169000}; MAXSEG=${MAXSEG:-3}
RUNS=/work/classic/fr_ml1145-muffto/runs/czm
source ~/software/muffto_env_gpu.sh
export DEVICE=gpu CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-$CUDA_ROOT}} XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
export K_RESIDUAL=${K_RESIDUAL:-1e-5} MAX_STAGGERED=${MAX_STAGGERED:-500}
cd ~/software/muFFTTO/experiments/dual_preconditioner/localization_softening_bands
echo "job $SLURM_JOB_ID host $(hostname) partition $SLURM_JOB_PARTITION start $(date -Is) specs: $* SEG=$SEG RESTART_FROM=${RESTART_FROM:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
( sleep 120; while true; do echo "$(date +%H:%M) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ') util $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)"; sleep 600; done ) &
mon=$!
rc_all=0
for spec in "$@"; do
  IFS=: read -r D N P DIR <<< "$spec"
  K=$(python3 -c "print(f'{float(\"$K_RESIDUAL\"):g}')")
  STEM=PFCZM${D}D_N${N}_${P}_${DIR}_b1.5mm_k${K}${RUN_SUFFIX:-}
  SUF=""; if [ "$SEG" -gt 0 ]; then SUF=_r$SEG; fi
  LOG=$RUNS/logs/${STEM}${SUF}.log
  start=$(date +%s); rc=0
  if [ $# -eq 1 ]; then TB=$BUDGET; else TB=1e12; fi
  env DIM=$D NPIX=$N LOAD_DIR=$DIR TIME_BUDGET=$TB ${RESTART_FROM:+RESTART_FROM=$RESTART_FROM} \
      python -u pfczm_concrete.py "$P" > "$LOG" 2>&1 || rc=$?
  echo "$(date -Is) $spec exit $rc wall $(( $(date +%s) - start )) s log $LOG"
  if [ "$rc" -eq 3 ] && [ $# -eq 1 ]; then
    if [ "$SEG" -ge $((MAXSEG - 1)) ]; then echo "time budget, MAXSEG=$MAXSEG reached: not continuing"; kill $mon; exit 1; fi
    NEXT=$(sbatch --parsable -p "$SLURM_JOB_PARTITION" --mem="${SLURM_MEM_PER_NODE}M" --job-name="${SLURM_JOB_NAME}_r$((SEG + 1))" \
           --export=ALL,SEG=$((SEG + 1)),RESTART_FROM=$PWD/exp_data/${STEM}_checkpoint.npz sbatch_czm_gpu.sh "$spec")
    echo "time budget: continued as job $NEXT (segment $((SEG + 1)))"
    echo "$NEXT $spec segment $((SEG + 1)) (continues $SLURM_JOB_ID)" >> $RUNS/jobs.txt
    kill $mon; exit 0
  fi
  if [ "$rc" -ne 0 ]; then rc_all=$rc; fi
done
kill $mon 2>/dev/null || true
echo "done $(date -Is)"
exit $rc_all

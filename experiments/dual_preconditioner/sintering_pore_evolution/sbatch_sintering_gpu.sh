#!/bin/bash
# One sintering run (sintering_phase_field.py) on one NEMO2 GPU:
#   sbatch [-p l40s] sbatch_sintering_gpu.sh <N> <Green|Green_Jacobi|dual> <WIDTH_nm>
# Same settings as run_sintering_study.sh (dt 1e-7 s, 5000 steps, mechanics every 25, snapshots every 50, P 50 MPa).
# exp_data -> /work/classic/fr_ml1145-muffto/runs/sintering/exp_data (symlink). 48 h limit: stopped after BUDGET s,
# then resubmits itself from the checkpoint (CHECKPOINT_SECONDS 1800), at most MAXSEG (3) segments.
#SBATCH --job-name=SINT_gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --exclude=g3606,g3608
#SBATCH --output=/work/classic/fr_ml1145-muffto/runs/sintering/slurm/SINT_gpu-%j.out
N=$1; P=$2; W=$3
SEG=${SEG:-0}; BUDGET=${BUDGET:-171000}; MAXSEG=${MAXSEG:-3}
R=/work/classic/fr_ml1145-muffto/runs/sintering
cd ~/software/muFFTTO/experiments/dual_preconditioner/sintering_pore_evolution
source ~/software/muffto_env_gpu.sh
export DEVICE=gpu CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-$CUDA_ROOT}} XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg OMP_NUM_THREADS=2 JAX_PLATFORMS=cpu
SUF=$([ "$W" = 4 ] && echo "" || echo "_W$W")
export NPIX=$N WIDTH=$W RUN_SUFFIX=$SUF DT=${DT:-1e-7} STEPS=${STEPS:-5000} MECH_EVERY=${MECH_EVERY:-25} SNAP=${SNAP:-50} CHECKPOINT_SECONDS=${CHECKPOINT_SECONDS:-1800}
LOG=$R/logs/SINT_N${N}_${P}_W${W}$([ "$SEG" -gt 0 ] && echo "_r$SEG").log
echo "job $SLURM_JOB_ID host $(hostname) start $(date -Is) N=$N $P W=$W seg=$SEG RESTART_FROM=${RESTART_FROM:-}"
start=$(date +%s); rc=0
timeout -s TERM -k 120 "$BUDGET" python -u sintering_phase_field.py "$P" > "$LOG" 2>&1 || rc=$?
echo "done $(date -Is) exit $rc wall $(( $(date +%s) - start )) s"
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  [ "$SEG" -ge $((MAXSEG - 1)) ] && { echo "MAXSEG reached"; exit 1; }
  FROM=$(ls -t exp_data/SINT_N${N}_${P}_W${W}_*${SUF}_checkpoint.npz 2>/dev/null | head -1)
  [ -z "$FROM" ] && { echo "no checkpoint"; exit 1; }
  NEXT=$(sbatch --parsable -p "$SLURM_JOB_PARTITION" --job-name="${SLURM_JOB_NAME}_r$((SEG + 1))" \
         --export=ALL,SEG=$((SEG + 1)),RESTART_FROM=$(readlink -f $FROM) sbatch_sintering_gpu.sh "$N" "$P" "$W")
  echo "$NEXT N=$N $P W=$W segment $((SEG + 1))" >> $R/jobs.txt
fi

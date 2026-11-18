#!/usr/bin/env bash
# to3d_large.py on several GPUs of one NEMO node, one MPI rank per GPU (run inside an sbatch job, see fable_job.sh):
#   sbatch -J FB_x -p h200 --gres=gpu:4 --ntasks=4 ... fable_job.sh bash fable_run_TO_mpi.sh <tag> <N> [MAXITER, default 10]
# Reference setup as fable_run_TO.sh (extra settings through the environment: WARM_STORE, SYMMETRY, INIT_FROM, ...).
# Samples the memory of every GPU of the job -> logs/fable/<tag>_N<N>_np<ranks>.log, one line in logs/fable/peak.txt
cd "$(dirname "$0")"; source ./fable_env.sh
TAG=$1; N=$2; export MAXITER=${3:-10}; NP=${SLURM_NTASKS:-1}
LOG=logs/fable/${TAG}_N${N}_np$NP.log; mkdir -p logs/fable
export ELEMENT=${ELEMENT:-trilinear_hexahedron_1Q_hg} HG_EPS=${HG_EPS:-0.3} DEVICE=gpu NPIX=$N \
       ETA=$($PY -c "print(2 / $N)") SOFT_EXP=3 PF_WEIGHT=0.2 CG_DELAY=0 GTOL=0 OBJECTIVE=energy TARGET_NU=-0.5 \
       TARGET_SCALE=0.3 HISTORY_EVERY=${HISTORY_EVERY:-10} HELM_C=4 \
       INIT_FROM=${INIT_FROM:-$PWD/data/CH3D/CH3D_start_bi_s11.nc} RUN_SUFFIX=_fable_${TAG}_np$NP \
       OUT_DIR=${OUT_DIR:-/work/classic/fr_ml1145-muffto/runs/FABLE/data}
srun --ntasks=$NP --gpus-per-task=1 --cpus-per-task=${SLURM_CPUS_PER_TASK:-4} $PY -u to3d_large.py > "$LOG" 2>&1 & P=$!; M=0
while kill -0 $P 2>/dev/null; do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ -n "$U" ] && [ "$U" -gt "$M" ] && M=$U
  sleep 1
done
wait $P; RC=$?
echo "$TAG N $N np $NP peak $M MiB on the fullest GPU, exit $RC $(grep -c '^EVAL' "$LOG") evals $(grep '^TOTAL' "$LOG")" >> logs/fable/peak.txt

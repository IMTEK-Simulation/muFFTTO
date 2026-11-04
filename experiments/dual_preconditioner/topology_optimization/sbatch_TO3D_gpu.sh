#!/bin/bash
# 3D phase-field TO (example_3D_elasticity_TO.py) on one L40S GPU of NEMO2, contrast 1e4 and w = 0.2 by default.
#   sbatch [--job-name=..] sbatch_TO3D_gpu.sh <N> <precond> <eta> [segment=1] [iterations_done=0]
#   env: SOFT_EXP (default 4), PF_WEIGHT (default 0.2), HISTORY_EVERY (default 10), BUDGET (s, default 47 h 30 min)
# 48 h limit: the run is stopped after BUDGET; if the optimizer has not finished, the job submits segment k+1, started
# (INIT_FROM) from the last iterate in segment k's _history.nc, with the rest of the 500 iterations (MAXITER); at most
# MAXSEG (default 4) segments. L-BFGS memory restarts at each segment boundary.
# Logs: $MUFFTTO_WS/runs/TO3D/logs/TO3D_N<N>_eta<eta>_<precond>[_seg<k>].log (+ slurm .out); data in
# topology_optimization/data/example_3D_elasticity_TO/ (see NOTES_eta_h_choice.md)
#SBATCH --job-name=TO3D_gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --output=/work/classic/fr_ml1145-muffto/runs/TO3D/logs/slurm-%j.out
N=$1; P=$2; ETA=$3; SEG=${4:-1}; DONE=${5:-0}
MAXSEG=${MAXSEG:-4}; BUDGET=${BUDGET:-171000}; HE=${HISTORY_EVERY:-10}
cd ~/software/muFFTTO/experiments/dual_preconditioner/topology_optimization
source ~/software/muffto_env_gpu.sh
export DEVICE=gpu CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-$CUDA_ROOT}} OMP_NUM_THREADS=1
SUF=${BASE_SUFFIX:-}$([ "$SEG" -gt 1 ] && echo "_seg$SEG")
LOG=$MUFFTTO_WS/runs/TO3D/logs/TO3D_N${N}_eta${ETA}_${P}${SUF}.log
mkdir -p "$(dirname "$LOG")"
echo "start $(date -Is) host $(hostname) job $SLURM_JOB_ID N=$N P=$P ETA=$ETA segment=$SEG iterations_done=$DONE" \
     "INIT_FROM=${INIT_FROM:-}" > "$LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> "$LOG"
start=$(date +%s)
RUN_SUFFIX=$SUF MAXITER=$((500 - DONE)) NPIX=$N SOFT_EXP=${SOFT_EXP:-4} ETA=$ETA CG_MODE=energy GTOL=0 \
HISTORY_EVERY=$HE PF_WEIGHT=${PF_WEIGHT:-0.2} \
    timeout -s TERM -k 120 "$BUDGET" python -u example_3D_elasticity_TO.py "$P" >> "$LOG" 2>&1
rc=$?
echo "WALL $(( $(date +%s) - start )) s exit $rc" >> "$LOG"
grep -q "Data saved" "$LOG" && exit 0
[ "$rc" -ne 124 ] && [ "$rc" -ne 137 ] && { echo "segment $SEG failed (exit $rc), not continuing" >> "$LOG"; exit 1; }
[ "$SEG" -ge "$MAXSEG" ] && { echo "MAXSEG=$MAXSEG reached, not continuing" >> "$LOG"; exit 1; }
HIST=$(ls -t data/example_3D_elasticity_TO/N${N}_*_${P}_eta_${ETA}_pfw*_l_bfgs${SUF}_history.nc 2>/dev/null | head -n 1)
FRAMES=$(python -c "import netCDF4,sys; print(len(netCDF4.Dataset(sys.argv[1])['phase_field_iterate']))" "$HIST" 2>/dev/null)
[ -z "$FRAMES" ] || [ "$FRAMES" -lt 1 ] && { echo "no saved iterate found, not continuing" >> "$LOG"; exit 1; }
NEW_DONE=$(( DONE + HE * (FRAMES - 1) + 1 ))          # frames are saved at iterations 1, 1+HE, 1+2HE, ...
NEXT=$(sbatch --parsable --job-name="${SLURM_JOB_NAME}_s$((SEG + 1))" --export=ALL,INIT_FROM=$PWD/$HIST \
       sbatch_TO3D_gpu.sh "$N" "$P" "$ETA" $((SEG + 1)) "$NEW_DONE")
echo "continued: job $NEXT, segment $((SEG + 1)) from $HIST (iteration $NEW_DONE of 500)" >> "$LOG"

#!/bin/bash
# 3D phase-field TO (example_3D_elasticity_TO.py) on one L40S GPU of NEMO2, contrast 1e4 and w = 0.2 by default.
#   sbatch [--job-name=..] sbatch_TO3D_gpu.sh <N> <precond> <eta> [segment=1] [iterations_done=0]
#   env: SOFT_EXP (default 4), PF_WEIGHT (default 0.2), HISTORY_EVERY (default 10), BUDGET (s, default 47 h 30 min)
#        LOW_MEMORY (default 1 for N >= 192: the grid does not fit on the GPU otherwise; Discretization(low_memory=))
# Runs until the optimizer converges: every job (segment) does at most SEG_ITERS (default 500) L-BFGS iterations and
# saves its design (_final.nc, suffix _seg<k> from k = 2). If L-BFGS stopped on the iteration cap (success=False), the
# job submits segment k+1 started (INIT_FROM) from that design; after a 48 h cut (BUDGET) from the last iterate of the
# _history.nc. It stops when L-BFGS converges (xtol: largest change of the phase < 1e-3) or after MAXSEG segments
# (default 10). L-BFGS memory restarts at each segment boundary.
#   continue a finished run: INIT_FROM=<its _final.nc> sbatch ... sbatch_TO3D_gpu.sh <N> <P> <eta> 2 <iterations done>
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
#SBATCH --exclude=g3606,g3608         # GPU ECC errors (killed TO 128 Green and two 64^3 continuations)
#SBATCH --output=/work/classic/fr_ml1145-muffto/runs/TO3D/logs/slurm-%j.out
N=$1; P=$2; ETA=$3; SEG=${4:-1}; DONE=${5:-0}
MAXSEG=${MAXSEG:-10}; SEG_ITERS=${SEG_ITERS:-500}; BUDGET=${BUDGET:-171000}; HE=${HISTORY_EVERY:-10}
cd ~/software/muFFTTO/experiments/dual_preconditioner/topology_optimization
source ~/software/muffto_env_gpu.sh
export DEVICE=gpu CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-$CUDA_ROOT}} OMP_NUM_THREADS=1
SUF=${BASE_SUFFIX:-}$([ "$SEG" -gt 1 ] && echo "_seg$SEG")
SW=${STRESS_WEIGHT:-1}; SWTAG=$([ "$SW" != 1 ] && echo "_sw$SW")   # as in the data file names (two weights never share a log)
LOG=$MUFFTTO_WS/runs/TO3D/logs/TO3D_N${N}_eta${ETA}_${P}${SWTAG}${SUF}.log
mkdir -p "$(dirname "$LOG")"
echo "start $(date -Is) host $(hostname) job $SLURM_JOB_ID N=$N P=$P ETA=$ETA segment=$SEG iterations_done=$DONE" \
     "INIT_FROM=${INIT_FROM:-}" > "$LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> "$LOG"
start=$(date +%s)
RUN_SUFFIX=$SUF MAXITER=$SEG_ITERS NPIX=$N SOFT_EXP=${SOFT_EXP:-4} ETA=$ETA CG_MODE=energy GTOL=0 \
HISTORY_EVERY=$HE PF_WEIGHT=${PF_WEIGHT:-0.2} LOW_MEMORY=${LOW_MEMORY:-$([ "$N" -ge 192 ] && echo 1 || echo 0)} \
    timeout -s TERM -k 120 "$BUDGET" python -u example_3D_elasticity_TO.py "$P" >> "$LOG" 2>&1
rc=$?
echo "WALL $(( $(date +%s) - start )) s exit $rc" >> "$LOG"
if grep -q "OPTIMIZER success=True" "$LOG"; then
    echo "converged after $(( DONE + $(grep -oP 'OPTIMIZER success=True nit=\K[0-9]+' "$LOG") )) iterations in total" >> "$LOG"; exit 0
fi
[ "$SEG" -ge "$MAXSEG" ] && { echo "MAXSEG=$MAXSEG reached, not continuing" >> "$LOG"; exit 1; }
if grep -q "Data saved" "$LOG"; then                     # iteration cap (or failed line search): continue from the design
    START=$(ls -t data/example_3D_elasticity_TO/N${N}_*_${P}_eta_${ETA}_pfw*${SWTAG}_l_bfgs${SUF}_final.nc 2>/dev/null | head -n 1)
    NIT=$(grep -oP 'OPTIMIZER success=False nit=\K[0-9]+' "$LOG" | tail -n 1)
    [ -z "$START" ] || [ -z "$NIT" ] && { echo "no final design / iteration count found, not continuing" >> "$LOG"; exit 1; }
    NEW_DONE=$(( DONE + NIT ))
elif [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then      # time budget: continue from the last stored iterate
    START=$(ls -t data/example_3D_elasticity_TO/N${N}_*_${P}_eta_${ETA}_pfw*${SWTAG}_l_bfgs${SUF}_history.nc 2>/dev/null | head -n 1)
    FRAMES=$(python -c "import netCDF4,sys; print(len(netCDF4.Dataset(sys.argv[1])['phase_field_iterate']))" "$START" 2>/dev/null)
    [ -z "$FRAMES" ] || [ "$FRAMES" -lt 1 ] && { echo "no saved iterate found, not continuing" >> "$LOG"; exit 1; }
    NEW_DONE=$(( DONE + HE * (FRAMES - 1) + 1 ))      # frames are saved at iterations 1, 1+HE, 1+2HE, ...
else
    echo "segment $SEG failed (exit $rc), not continuing" >> "$LOG"; exit 1
fi
NEXT=$(sbatch --parsable -p "$SLURM_JOB_PARTITION" --mem="${SLURM_MEM_PER_NODE}M" --job-name="${SLURM_JOB_NAME%_s[0-9]*}_s$((SEG + 1))" --export=ALL,INIT_FROM=$PWD/$START \
       sbatch_TO3D_gpu.sh "$N" "$P" "$ETA" $((SEG + 1)) "$NEW_DONE")
echo "continued: job $NEXT, segment $((SEG + 1)) from $START (iteration $NEW_DONE)" >> "$LOG"

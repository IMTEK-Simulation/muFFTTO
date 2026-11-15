#!/usr/bin/env bash
# One 3D TO run in the reference setup of the large-3D-TO study (FABLE_BRIEF.md section 2): 1Q_hg element (eps 0.3),
# MG-CG, 6 load cases, energy objective (nu* -0.5, scale 0.3), eta = 2h, contrast 1e3, Helmholtz L-BFGS (c = 4),
# start = 3D Cahn-Hilliard bi_s11. Samples the peak GPU memory of the process itself (nvidia-smi compute apps).
#   ./fable_run_TO.sh <tag> <N> [MAXITER, default 10]      extra settings through the environment (SCRIPT, SYMMETRY, ...)
#   -> logs/fable/<tag>_N<N>.log, one line in logs/fable/peak.txt
cd "$(dirname "$0")"; source ./fable_env.sh
TAG=$1; N=$2; export MAXITER=${3:-10}
LOG=logs/fable/${TAG}_N$N.log; mkdir -p logs/fable
export ELEMENT=${ELEMENT:-trilinear_hexahedron_1Q_hg} HG_EPS=${HG_EPS:-0.3} DEVICE=${DEVICE:-gpu} NPIX=$N \
       ETA=$($PY -c "print(2 / $N)") SOFT_EXP=3 PF_NORMALIZED=1 PF_WEIGHT=0.2 CG_MODE=energy CG_DELAY=0 GTOL=0 \
       OBJECTIVE=${OBJECTIVE:-energy} TARGET_NU=-0.5 TARGET_SCALE=0.3 HISTORY_EVERY=${HISTORY_EVERY:-10} SHOW_PLOTS=0 DESIGN_OPT=${DESIGN_OPT:-lbfgs_helm} HELM_C=4 \
       INIT_FROM=${INIT_FROM:-$PWD/data/CH3D/CH3D_start_bi_s11.nc} RUN_SUFFIX=_fable_$TAG
[ -n "$SLURM_JOB_ID" ] && export OUT_DIR=${OUT_DIR:-/work/classic/fr_ml1145-muffto/runs/FABLE/data}   # NEMO: workspace
$PY -u ${SCRIPT:-example_3D_elasticity_TO.py} MG > "$LOG" 2>&1 & P=$!; M=0
while kill -0 $P 2>/dev/null; do
  U=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$P '$1==p {print $2}')
  [ -n "$U" ] && [ "$U" -gt "$M" ] && M=$U
  sleep 0.3
done
wait $P; RC=$?
echo "$TAG N $N peak $M MiB exit $RC $(grep -c '^EVAL' "$LOG") evals $(grep '^TOTAL' "$LOG")" >> logs/fable/peak.txt

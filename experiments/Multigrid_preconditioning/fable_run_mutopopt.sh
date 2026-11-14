#!/usr/bin/env bash
# One muTopOpt run of the comparison problem (inside a job, one GPU): the problem of to3d_large.py with
# OBJECTIVE=stress -- solid K 1, G 0.5, void 1e-3, SIMP p 2, nodal design, target nu -0.5 at scale 0.3, interface
# width 2 voxels, interface weight 0.2 -- from muTopOpt's default start, up to MAXITER outer iterations.
#   fable_run_mutopopt.sh <tag> <N> <double|single> <trust_region|lbfgs> [MAXITER, default 500]
#   -> logs/fable/mutopopt_<tag>_N<N>.log, design $WS/runs/FABLE/data/mutopopt_<tag>_N<N>.nc, line in logs/fable/peak.txt
cd "$(dirname "$0")"; source ./fable_env.sh; source ./fable_mutopopt_env.sh
TAG=$1; N=$2; PRECISION=$3; METHOD=$4; MAXITER=${5:-500}
LOG=$PWD/logs/fable/mutopopt_${TAG}_N$N.log; OUT=/work/classic/fr_ml1145-muffto/runs/FABLE/data
CONTROL=$OUT/mutopopt_${TAG}_N$N.json; mkdir -p logs/fable $OUT
cat > $CONTROL <<JSON
{
  "grid": {"nb_grid_pts": [$N, $N, $N]},
  "material": {"young_modulus": 1.2857142857142858, "poisson_ratio": 0.2857142857142857,
               "simp_exponent": 2.0, "void_ratio": 1e-3},
  "target": {"bulk_modulus": 0.009642857142857142, "shear_modulus": 0.05785714285714286},
  "regularization": {"interface_width": $($PY -c "print(2 / $N)"), "weight": 0.2},
  "optimizer": {"method": "$METHOD", "max_iterations": $MAXITER},
  "compute": {"device": "gpu", "precision": "$PRECISION"},
  "output": {"file": "$OUT/mutopopt_${TAG}_N$N.nc", "restart_control": "$OUT/mutopopt_${TAG}_N$N.restart.json"}
}
JSON
START=$(date +%s)
$PY -u $MUTOPOPT/simulate.py $CONTROL > "$LOG" 2>&1 & P=$!; M=0
while kill -0 $P 2>/dev/null; do
  U=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$P '$1==p {print $2}')
  [ -n "$U" ] && [ "$U" -gt "$M" ] && M=$U
  sleep 0.5
done
wait $P; RC=$?
echo "mutopopt_$TAG N $N peak $M MiB exit $RC wall $(( $(date +%s) - START )) s $(grep '^done:' "$LOG" | cut -c1-200)" >> logs/fable/peak.txt

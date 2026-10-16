#!/usr/bin/env bash
# Grid study of the elasticity runs: P1-P3 x {green, green_jacobi, dual}, kappa = 1e-3, on 512^2 and 1024^2
# (KAPPA=3 PROBLEMS="P1 P2": kappa = 1e3 for the given problems)
# (256^2 comes from run_all_elasticity.sh; GRIDS="32 64 128" ./run_grids_elasticity.sh for coarser grids).
#   PYTHON=... ./run_grids_elasticity.sh [-j jobs, default 6]
set -u
cd "$(dirname "$0")"
JOBS=6
[[ $# -ge 2 && "$1" == "-j" ]] && JOBS="$2"
PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
mkdir -p logs_elasticity
{
for N in ${GRIDS:-512 1024}; do
    for prec in dual green green_jacobi; do
        for prob in ${PROBLEMS:-P1 P2 P3}; do echo "$prob ${KAPPA:--3} $N $prec"; done
    done
done
} | xargs -P "$JOBS" -L 1 bash -c '
    log="logs_elasticity/$0_N$2_kappa1e$1_$3.log"
    "'"$PYTHON"'" run_experiment_elasticity.py --problem "$0" --kappa-exp "$1" --grid "$2" --preconditioner "$3" \
        > "$log" 2>&1 && tail -n 1 "$log" || echo "FAILED: $log"'

#!/usr/bin/env bash
# All runs of the paper-1 preconditioner comparison: P1-P3 x {green, green_jacobi, dual},
# the paper's contrasts on 256^2.  The grid study (512^2, 1024^2, kappa = 1e-3) only with --grids.
#   ./run_all.sh            [-j parallel jobs, default 6]   [--grids]   [PYTHON=... ./run_all.sh]
set -u
cd "$(dirname "$0")"
JOBS=6
GRID_STUDY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -j) JOBS="$2"; shift 2 ;;
        --grids) GRID_STUDY=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS=1
export JAX_PLATFORMS=cpu          # CPU runs; otherwise every process preallocates GPU memory
mkdir -p logs
{
for prec in green green_jacobi dual; do
    for k in -1 -2 -3 1 2 3; do
        for prob in P1 P2; do echo "$prob $k 256 $prec"; done
    done
    for k in -1 -2 -3; do echo "P3 $k 256 $prec"; done
    if [[ $GRID_STUDY -eq 1 ]]; then
        for N in 512 1024; do
            for prob in P1 P2 P3; do echo "$prob -3 $N $prec"; done
        done
    fi
done
} | xargs -P "$JOBS" -L 1 bash -c '
    log="logs/$0_N$2_kappa1e$1_$3.log"
    "'"$PYTHON"'" run_experiment.py --problem "$0" --kappa-exp "$1" --grid "$2" --preconditioner "$3" > "$log" 2>&1 \
        && tail -n 1 "$log" || echo "FAILED: $log"'

#!/bin/bash
# linear CG-vs-BB sweep: every line of jobs_linear.txt is "<problem> <kappa_exp> <preconditioner>", N = 128
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
run_one() {
    log=logs_linear/$1_k$2_$3.log
    [ -s "$log" ] && grep -q 'k(1e-6)' "$log" && return
    OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu systemd-run --user --wait --quiet --unit=bbl-$1-k$2-$3 --same-dir \
        -p MemoryMax=4G -p RuntimeMaxSec=7200 -E OMP_NUM_THREADS=1 -E JAX_PLATFORMS=cpu \
        bash -c "$PY linear_P123.py --problem $1 --kappa-exp $2 --preconditioner $3 --grid 128 > $log 2>&1"
}
export -f run_one
export PY
xargs -P "${1:-3}" -L 1 bash -c 'run_one "$@"' _ < jobs_linear.txt

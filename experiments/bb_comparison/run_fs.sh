#!/bin/bash
# finite-strain Newton vs BB sweep: lines of jobs_fs.txt are "<geometry> <kappa_exp> <N> <preconditioner> <solver>"
cd "$(dirname "$0")"
mkdir -p logs_fs
PY=~/micromamba/envs/muffto-gpu/bin/python
run_one() {
    log=logs_fs/$1_k$2_N$3_$4_$5.log
    grep -q 'Newton .*BB' "$log" 2>/dev/null && return
    systemd-run --user --wait --quiet --unit=bbf-$1-k$2-N$3-$4-$5 --same-dir -p MemoryMax=4G -p RuntimeMaxSec=5400 \
        -E OMP_NUM_THREADS=1 -E JAX_PLATFORMS=cpu \
        bash -c "$PY -u finite_strain_P123.py -g $1 -k $2 -n $3 -p $4 --solver $5 > $log 2>&1"
}
export -f run_one
export PY
xargs -P "${1:-3}" -L 1 bash -c 'run_one "$@"' _ < jobs_fs.txt

#!/bin/bash
# BB comparison queue: every line of the job file is "<name> <script> VAR=value ...", run with systemd-run (MemoryMax,
# 2 h limit), JOBS in parallel; logs in logs/<name>.log. usage: ./run_queue.sh jobs.txt [JOBS]
cd "$(dirname "$0")"
mkdir -p logs
PY=~/micromamba/envs/muffto-gpu/bin/python
run_one() {
    name=$1; script=$2; shift 2
    [ -e logs/$name.done ] && return
    args=()
    for e in "$@" JAX_PLATFORMS=cpu OMP_NUM_THREADS=2; do args+=(-E "$e"); done
    systemd-run --user --wait --quiet --unit=bbq-$name --same-dir -p MemoryMax=6G -p RuntimeMaxSec=7200 "${args[@]}" \
        bash -c "$PY $script > logs/$name.log 2>&1"
    touch logs/$name.done
}
export -f run_one
export PY
grep -v '^#' "$1" | grep . | xargs -P "${2:-6}" -L 1 bash -c 'run_one "$@"' _

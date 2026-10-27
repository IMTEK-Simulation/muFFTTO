#!/bin/bash
# corrected Green with S solved inexactly by an inner (P)CG: inner tolerance / fixed inner steps, with and without the
# circulant wall preconditioner (INNER_PRECOND=circ); plain outer CG. Beam with 4 holes, void 0.00625, clamped face.
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
mkdir -p exp_data/logs
jobs() {
    for n in 32 64 128 256; do
        echo "$n PRECOND=green_dirichlet"
        for t in 1e-8 1e-6 1e-4 1e-2; do
            echo "$n PRECOND=green_dirichlet_inner INNER_TOL=$t"
            echo "$n PRECOND=green_dirichlet_inner INNER_TOL=$t INNER_PRECOND=circ"
        done
        for k in 5 10 20 40; do echo "$n PRECOND=green_dirichlet_inner INNER_MAXIT=$k"; done
        for k in 1 2 4 8; do echo "$n PRECOND=green_dirichlet_inner INNER_MAXIT=$k INNER_PRECOND=circ"; done
    done
}
run_one() {
    n=$1; shift
    name=inexact_n${n}_$(echo "$@" | tr ' =' '_-')
    systemd-run --user --wait --quiet --unit=inx-$name --same-dir -p MemoryMax=4G -p RuntimeMaxSec=3600 \
        -E JAX_PLATFORMS=cpu -E OMP_NUM_THREADS=1 -E OPENBLAS_NUM_THREADS=1 -E MKL_NUM_THREADS=1 -E SOLVE_ONLY=1 \
        -E HOLES=4 -E HOLE_RADIUS=0.05 -E VOID_THICKNESS=0.00625 -E CG_MAXITER=5000 -E N_H=$n $(printf -- '-E %s ' "$@") \
        bash -c "$PY -u cantilever_mesh_study.py > exp_data/logs/$name.txt 2>&1"
}
export -f run_one
export PY
jobs | xargs -P "${1:-6}" -L 1 bash -c 'run_one "$@"' _

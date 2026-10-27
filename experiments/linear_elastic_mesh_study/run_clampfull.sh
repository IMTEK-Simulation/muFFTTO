#!/bin/bash
# corrected Green with the clamp column through the whole cell height (CLAMP_FULL=1): S exactly block-circulant, solved
# directly by an FFT along the wall (INNER_DIRECT=1); compared with the assembled dense S and with plain inner CG
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
mkdir -p exp_data/logs
for n in 32 64 128 256; do
    echo "$n PRECOND=green_dirichlet"
    echo "$n PRECOND=green_dirichlet_inner INNER_PRECOND=circ INNER_DIRECT=1"
    echo "$n PRECOND=green_dirichlet_inner INNER_TOL=1e-8"
done | xargs -P "${1:-4}" -L 1 bash -c '
    n=$1; shift; name=clampfull_n${n}_$(echo "$@" | tr " =" "_-")
    systemd-run --user --wait --quiet --unit=cf-$name --same-dir -p MemoryMax=4G -p RuntimeMaxSec=3600 \
        -E JAX_PLATFORMS=cpu -E OMP_NUM_THREADS=1 -E OPENBLAS_NUM_THREADS=1 -E MKL_NUM_THREADS=1 -E SOLVE_ONLY=1 \
        -E HOLES=4 -E HOLE_RADIUS=0.05 -E VOID_THICKNESS=0.00625 -E CLAMP_FULL=1 -E N_H=$n $(printf -- "-E %s " "$@") \
        bash -c "'"$PY"' -u cantilever_mesh_study.py > exp_data/logs/$name.txt 2>&1"' _

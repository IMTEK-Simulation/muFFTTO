#!/bin/bash
# corrected Green with S solved by an inner CG (PRECOND=green_dirichlet_inner), beam with holes, void 0.00625
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SOLVE_ONLY=1 HOLES=4 HOLE_RADIUS=0.05 \
       VOID_THICKNESS=0.00625 PRECOND=green_dirichlet_inner
mkdir -p exp_data/logs
for n in 256 128 64 32; do for t in 1e-8 1e-4 1e-2; do echo "$t $n"; done; done |
    xargs -P 6 -n 2 sh -c 'INNER_TOL=$0 N_H=$1 '"$PY"' -u cantilever_mesh_study.py > exp_data/logs/inner$0_n$1.txt 2>&1'

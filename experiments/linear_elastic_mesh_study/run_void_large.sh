#!/bin/bash
# Void 0.00625 (beam with 4 holes) on large grids, n_H = 256, 512, 1024: clamped face with plain and corrected Green,
# forces only; 4 workers (largest first), stored runs are skipped. Then the figure.
# usage:  ./run_void_large.sh
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SOLVE_ONLY=1 HOLES=4 HOLE_RADIUS=0.05 \
       VOID_THICKNESS=0.00625
mkdir -p exp_data/logs
for n in 1024 512 256; do
    for case in "green_dirichlet clamp" "green clamp" "green traction"; do echo "$case $n"; done; done |
    xargs -P 4 -n 3 sh -c 'PRECOND=$0 BC=$1 N_H=$2 '"$PY"' -u cantilever_mesh_study.py > exp_data/logs/voidlarge_$0_$1_n$2.txt 2>&1'
$PY void_holes_study.py

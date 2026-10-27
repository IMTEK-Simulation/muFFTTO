#!/bin/bash
# Beam with 4 holes (radius 0.05) on its centre line, void 0.4, n_H = 16 ... 128: clamped face and pin + roller with
# plain and corrected Green, and the beam loaded by forces only; 6 workers, stored runs are skipped. Then the figure.
# usage:  ./run_holes.sh
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SOLVE_ONLY=1 VOID_THICKNESS=0.4 \
       HOLES=4 HOLE_RADIUS=0.05
mkdir -p exp_data/logs
for n in 128 64 32 16; do
    for case in "green clamp" "green_dirichlet clamp" "green simple" "green_dirichlet simple" "green traction"; do
        echo "$case $n"; done; done |
    xargs -P 6 -n 3 sh -c 'PRECOND=$0 BC=$1 N_H=$2 '"$PY"' -u cantilever_mesh_study.py > exp_data/logs/holes_$0_$1_n$2.txt 2>&1'
$PY holes_study.py

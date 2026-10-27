#!/bin/bash
# Thinner voids for the beam with 4 holes: 0.025 and 0.0125 (n_H = 16 ... 128) and 0.00625 (n_H = 32 ... 128, so that
# the void is at least one pixel); clamped face with plain and corrected Green, forces only. 6 workers. Then the figure.
# usage:  ./run_void_holes_thin.sh
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SOLVE_ONLY=1 HOLES=4 HOLE_RADIUS=0.05
mkdir -p exp_data/logs
for n in 128 64 32 16; do for t in 0.025 0.0125 0.00625; do
    [ "$t" = 0.00625 ] && [ $n -lt 32 ] && continue
    for case in "green clamp" "green_dirichlet clamp" "green traction"; do echo "$t $case $n"; done; done; done |
    xargs -P 6 -n 4 sh -c 'VOID_THICKNESS=$0 PRECOND=$1 BC=$2 N_H=$3 '"$PY"' -u cantilever_mesh_study.py > exp_data/logs/voidholes$0_$1_$2_n$3.txt 2>&1'
$PY void_holes_study.py

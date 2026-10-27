#!/bin/bash
# How thin may the void be? Beam with 4 holes, void 0.05 / 0.1 / 0.2 (0.4 is in run_holes.sh), n_H = 16 ... 128:
# clamped face with plain and corrected Green, and the beam loaded by forces only; 6 workers, stored runs skipped.
# usage:  ./run_void_holes.sh
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SOLVE_ONLY=1 HOLES=4 HOLE_RADIUS=0.05
mkdir -p exp_data/logs
for n in 128 64 32 16; do for t in 0.2 0.1 0.05; do
    for case in "green clamp" "green_dirichlet clamp" "green traction"; do echo "$t $case $n"; done; done; done |
    xargs -P 6 -n 4 sh -c 'VOID_THICKNESS=$0 PRECOND=$1 BC=$2 N_H=$3 '"$PY"' -u cantilever_mesh_study.py > exp_data/logs/voidholes$0_$1_$2_n$3.txt 2>&1'
$PY void_holes_study.py

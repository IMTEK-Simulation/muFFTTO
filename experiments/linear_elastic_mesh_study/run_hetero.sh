#!/bin/bash
# Beam with 4 holes and heterogeneous stiffness (contrast 10 and 100), void 0.4, n_H = 16 ... 128: clamped face with
# plain, corrected and S-only corrected Green, pin + roller with plain and corrected Green, forces only;
# 6 workers, stored runs are skipped. Then the figures.
# usage:  ./run_hetero.sh
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SOLVE_ONLY=1 VOID_THICKNESS=0.4 \
       HOLES=4 HOLE_RADIUS=0.05
mkdir -p exp_data/logs
for n in 128 64 32 16; do for k in 10 100; do
    for case in "green clamp" "green_dirichlet clamp" "green_dirichlet_S clamp" "green simple" \
                "green_dirichlet simple" "green traction"; do echo "$k $case $n"; done; done; done |
    xargs -P 6 -n 4 sh -c 'HETERO=$0 PRECOND=$1 BC=$2 N_H=$3 '"$PY"' -u cantilever_mesh_study.py > exp_data/logs/het$0_$1_$2_n$3.txt 2>&1'
for k in 10 100; do $PY holes_study.py --hetero $k; done

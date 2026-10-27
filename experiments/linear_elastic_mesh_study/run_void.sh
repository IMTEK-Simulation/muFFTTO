#!/bin/bash
# Void-thickness study: the cantilever with a void of fixed PHYSICAL thickness t on each side (instead of 2 pixels),
# on several meshes; 8 workers in parallel, stored runs are skipped. Then the figure.
# usage:  ./run_void.sh
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SOLVE_ONLY=1
mkdir -p exp_data/logs
for t in 0.025 0.05 0.1 0.2 0.4 0.8; do for n in 8 16 32 64; do echo "$t $n"; done; done |
    xargs -P 8 -n 2 sh -c 'VOID_THICKNESS=$0 N_H=$1 '"$PY"' -u cantilever_mesh_study.py > exp_data/logs/void$0_n$1.txt 2>&1'
$PY void_study.py

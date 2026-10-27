#!/bin/bash
# Supports study: clamp / simple / free / traction, void of fixed physical thickness 0.4 (no void effect), meshes n_H = 8 ... 128;
# 8 workers in parallel, stored runs are skipped. Then the figure.
# usage:  ./run_bc.sh
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SOLVE_ONLY=1 VOID_THICKNESS=0.4
mkdir -p exp_data/logs
for n in 128 64 32 16 8; do for bc in clamp simple free traction; do echo "$bc $n"; done; done |
    xargs -P 8 -n 2 sh -c 'BC=$0 N_H=$1 '"$PY"' -u cantilever_mesh_study.py > exp_data/logs/bc$0_n$1.txt 2>&1'
$PY bc_study.py

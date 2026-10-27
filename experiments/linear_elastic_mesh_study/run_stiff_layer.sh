#!/bin/bash
# Stiff layer of fixed physical thickness 0.05 (= H/4) instead of the clamped face, STIFF_FACTOR = 1 ... 1e4, held at
# 2 nodes; void 0.4; n_H = 16 ... 128; 6 workers, stored runs are skipped. Then the figure.
# usage:  ./run_stiff_layer.sh
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SOLVE_ONLY=1 VOID_THICKNESS=0.4 \
       BC=stiff STIFF_THICKNESS=0.05
mkdir -p exp_data/logs
for n in 128 64 32 16; do for k in 1 10 100 1000 10000; do echo "$k $n"; done; done |
    xargs -P 6 -n 2 sh -c 'STIFF_FACTOR=$0 N_H=$1 '"$PY"' -u cantilever_mesh_study.py > exp_data/logs/stiff_t0.05_$0_n$1.txt 2>&1'
$PY stiff_layer_study.py

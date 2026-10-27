#!/bin/bash
# 3D check of the theory: clamp / forces only x 2 void voxels / void 0.2, meshes n_H = 4 ... 32, one process per run
# (largest first, 6 at a time; stored runs are skipped), then the figure.
# usage:  ./run_3D.sh
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p exp_data/logs
for n in 32 16 8 4; do for bc in clamp traction; do for t in 0.2 0; do echo "$bc $t $n"; done; done; done |
    xargs -P 6 -n 3 sh -c 'BC=$0 VOID_THICKNESS=$1 N_H=$2 '"$PY"' -u cantilever_3D_study.py > exp_data/logs/3d_$0_t$1_n$2.txt 2>&1'
grep -h "n_H =" exp_data/logs/3d_*.txt
PLOT_ONLY=1 $PY cantilever_3D_study.py

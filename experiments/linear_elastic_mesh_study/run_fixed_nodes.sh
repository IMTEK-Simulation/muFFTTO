#!/bin/bash
# 1. number of fixed nodes: centred partial clamp of k = 3 ... n_H + 1 nodes, at n_H = 16, 32, 64
# 2. stiff layer instead of the clamp: one pixel column STIFF_FACTOR = 10 ... 1e4 times stiffer, fixed at 2 nodes,
#    at n_H = 16 ... 128
# void 0.4 everywhere (no void effect); 6 workers; stored runs are skipped. Then the figure.
# usage:  ./run_fixed_nodes.sh
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SOLVE_ONLY=1 VOID_THICKNESS=0.4
mkdir -p exp_data/logs
{
    for n in 128 64 32 16; do for k in 10 100 1000 10000; do echo "stiff $k 1 $n"; done; done
    for n in 64 32 16; do for nodes in 3 5 9 17 33 65; do
        [ $nodes -le $((n + 1)) ] && echo "clamp 0 $(python3 -c "print(($nodes - 1) / $n)") $n"; done; done
} | xargs -P 6 -n 4 sh -c 'BC=$0 STIFF_FACTOR=$1 CLAMP_SPAN=$2 N_H=$3 '"$PY"' -u cantilever_mesh_study.py > exp_data/logs/fixed_$0_$1_$2_n$3.txt 2>&1'
$PY fixed_nodes_study.py

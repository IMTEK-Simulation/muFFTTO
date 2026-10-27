#!/bin/bash
# Removing the effect of Dirichlet conditions, void 0.4, n_H = 16 ... 128 (6 workers, stored runs are skipped):
#   A. Green corrected to the constrained reference problem (PRECOND=green_dirichlet) for: clamped face, pin + roller,
#      partial clamps of 3 and 17 nodes, stiff layer (0.05 thick) held at 2 nodes
#   B. the stiff layer (0.05 thick) held only by its reaction forces (BC=stiffforce), plain Green
# then the figure.
# usage:  ./run_dirichlet_fix.sh
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SOLVE_ONLY=1 VOID_THICKNESS=0.4 \
       STIFF_THICKNESS=0.05
mkdir -p exp_data/logs
{
    for n in 128 64 32 16; do
        echo "green_dirichlet clamp 100 1 $n"
        echo "green_dirichlet simple 100 1 $n"
        for k in 100 10000; do echo "green_dirichlet stiff $k 1 $n"; echo "green stiffforce $k 1 $n"; done
        for k in 1 10 1000; do echo "green stiffforce $k 1 $n"; done
    done
    for n in 64 32 16; do for nodes in 3 17; do
        echo "green_dirichlet clamp 100 $(python3 -c "print(($nodes - 1) / $n)") $n"; done; done
} | xargs -P 6 -n 5 sh -c 'PRECOND=$0 BC=$1 STIFF_FACTOR=$2 CLAMP_SPAN=$3 N_H=$4 '"$PY"' -u cantilever_mesh_study.py > exp_data/logs/dfix_$0_$1_$2_$3_n$4.txt 2>&1'
$PY dirichlet_fix_study.py

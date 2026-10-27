#!/bin/bash
# Mesh study in parallel: one worker process per mesh (each stores exp_data/runs/n_H<n>_*.npz and skips meshes
# already stored), then the figures from the stored results.
# usage:  ./run_parallel.sh [2,4,8,16,32,64,128,256]
cd "$(dirname "$0")"
MESHES=${1:-2,4,8,16,32,64,128,256}
PY=~/micromamba/envs/muffto-gpu/bin/python
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p exp_data/logs
for n in ${MESHES//,/ }; do
    SOLVE_ONLY=1 N_H=$n $PY -u cantilever_mesh_study.py > exp_data/logs/n_H$n.txt 2>&1 &
done
wait
grep -h "^n_H" exp_data/logs/n_H*.txt | sort -k3 -n
PLOT_ONLY=1 N_H=$MESHES $PY cantilever_mesh_study.py 2>&1 | grep -E "order|extrap|wrote"

#!/bin/bash
# Plateau study: the cantilever at fixed meshes with one parameter varied at a time, all variants in parallel
# (stored runs are skipped), then the comparison figure per mesh.
# usage:  ./run_plateau.sh [32,64]
cd "$(dirname "$0")"
MESHES=${1:-32,64}
PY=~/micromamba/envs/muffto-gpu/bin/python
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SOLVE_ONLY=1 N_H=$MESHES
mkdir -p exp_data/logs
for v in 0.5 2 4; do BEAM_LENGTH=$v $PY -u cantilever_mesh_study.py > exp_data/logs/plateau_L$v.txt 2>&1 & done
for v in 2 4 8; do CLAMP_COLUMNS=$v $PY -u cantilever_mesh_study.py > exp_data/logs/plateau_cc$v.txt 2>&1 & done
for v in 0.5 0.25; do CLAMP_SPAN=$v $PY -u cantilever_mesh_study.py > exp_data/logs/plateau_span$v.txt 2>&1 & done
for v in 0.5 1 2; do BEAM_LENGTH=4 LOAD_X=$v $PY -u cantilever_mesh_study.py > exp_data/logs/plateau_L4_load$v.txt 2>&1 & done
wait
for n in ${MESHES//,/ }; do $PY plateau_study.py --n_H $n; done

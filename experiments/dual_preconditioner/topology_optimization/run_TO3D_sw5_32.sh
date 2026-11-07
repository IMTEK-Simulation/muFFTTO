#!/bin/bash
# 3D TO 32^3, eta = 4h = 1/8, stress weight a = 5, w = 0.2, Green-Jacobi, contrast 1e4, on the laptop GPU, one run
# after the other: standard sine start and the two 3D Cahn-Hilliard starts (make_TO3D_CH_starts.py).
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export NPIX=32 ETA=0.125 SOFT_EXP=4 CG_MODE=energy GTOL=0 MAXITER=3000 HISTORY_EVERY=10 STRESS_WEIGHT=5 DEVICE=gpu \
       MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
       CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux
for S in sine bi_s0 bi_s1; do
    init=$([ $S = sine ] || echo "INIT_FROM=$PWD/data/TO3D_CH_starts/CH3D_start_$S.nc")
    systemd-run --user --scope -q -p MemoryMax=12G env $init RUN_SUFFIX=_start_$S \
        $PY -u example_3D_elasticity_TO.py Green_Jacobi > logs/TO3D_sw5/N32_$S.log 2>&1
    echo "$S done $(date +%T)"
done

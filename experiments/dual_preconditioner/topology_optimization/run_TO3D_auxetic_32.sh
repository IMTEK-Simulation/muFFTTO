#!/bin/bash
# 3D TO toward strongly auxetic designs: compliant target (TARGET_SCALE x the standard target, nu = -0.5), 32^3,
# eta = 4h = 1/8, a = 5, w = 0.2, Green-Jacobi, contrast 1e4, from the 3D Cahn-Hilliard start bi_s1, laptop GPU.
#   ./run_TO3D_auxetic_32.sh <TARGET_SCALE> [MAXITER=3000]   -> logs/TO3D_auxetic/N32_ts<scale>.log
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
TS=$1; MI=${2:-3000}
export NPIX=32 ETA=0.125 SOFT_EXP=4 CG_MODE=energy GTOL=0 MAXITER=$MI HISTORY_EVERY=10 STRESS_WEIGHT=5 DEVICE=gpu \
       MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
       CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux TARGET_SCALE=$TS
systemd-run --user --scope -q -p MemoryMax=12G env INIT_FROM=$PWD/data/TO3D_CH_starts/CH3D_start_bi_s1.nc \
    RUN_SUFFIX=_start_bi_s1_ts$TS $PY -u example_3D_elasticity_TO.py Green_Jacobi > logs/TO3D_auxetic/N32_ts$TS.log 2>&1

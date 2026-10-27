#!/bin/bash
# AT2 1024^2 polydisperse pores, dual, GPU: continue the stopped run after its last converged load step (65), with
# checkpoints inside the load step every 100 staggered iterations; output *_restart.*
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu CUDA_PATH=~/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
MICROSTRUCTURE=polydisperse_pores DEVICE=gpu NPIX=1024 LENGTH_SCALE=0.03125 RUN_SUFFIX=_restart CHECKPOINT_EVERY=100 \
    RESTART_FROM=exp_data/AT2_N1024_dual_polydisperse_pores_l0.03125.nc RESTART_STEP=65 \
    ~/micromamba/envs/muffto-gpu/bin/python -u example_2D_phase_field_fracture_AT2.py dual \
    > logs/AT2_N1024_dual_polydisperse_pores_l0.03125_restart.log 2>&1

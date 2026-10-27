#!/bin/bash
# fixed-l AT2 on the polydisperse pore microstructure, 512^2, dual preconditioner, on the GPU (checkpoints every
# CHECKPOINT_EVERY=100 staggered iterations by default)
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu CUDA_PATH=~/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
MICROSTRUCTURE=polydisperse_pores DEVICE=gpu NPIX=512 LENGTH_SCALE=0.03125 ~/micromamba/envs/muffto-gpu/bin/python -u \
    example_2D_phase_field_fracture_AT2.py dual > logs/AT2_N512_dual_polydisperse_pores_l0.03125.log 2>&1

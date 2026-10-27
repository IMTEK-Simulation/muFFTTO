#!/bin/bash
# AT2, 1024^2, dual elastic preconditioner, on the GPU: single pore (paper setup) first, then the polydisperse pores
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu CUDA_PATH=~/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
for micro in circle_inclusion polydisperse_pores; do
    tag=$([ $micro = circle_inclusion ] && echo "" || echo "_$micro")
    MICROSTRUCTURE=$micro DEVICE=gpu NPIX=1024 LENGTH_SCALE=0.03125 ~/micromamba/envs/muffto-gpu/bin/python -u \
        example_2D_phase_field_fracture_AT2.py dual > logs/AT2_N1024_dual${tag}_l0.03125.log 2>&1
done

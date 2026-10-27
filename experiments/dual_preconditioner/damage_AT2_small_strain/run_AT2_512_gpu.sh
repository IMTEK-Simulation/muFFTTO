#!/bin/bash
# fixed-l AT2 series, 512^2, on the GPU, the three elastic preconditioners one after another
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu CUDA_PATH=~/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
for p in dual Green_Jacobi Green; do
    DEVICE=gpu NPIX=512 LENGTH_SCALE=0.03125 ~/micromamba/envs/muffto-gpu/bin/python -u example_2D_phase_field_fracture_AT2.py $p \
        > logs/AT2_N512_${p}_l0.03125.log 2>&1
done

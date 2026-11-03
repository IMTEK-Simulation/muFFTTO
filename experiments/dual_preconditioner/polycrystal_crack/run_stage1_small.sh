#!/bin/bash
# Stage 1, small scale: linear polycrystal + edge crack, CG iterations of Green / Green-Jacobi / dual
# against the grid (32^2 ... 128^2, i.e. ~2 ... 8.5 pixels per grain) and the crack/void stiffness k.
# CPU only (JAX_PLATFORMS=cpu keeps it off the GPU), serial, single thread.
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
mkdir -p logs
for N in 32 64 128; do
    for K in 1e-2 1e-4 1e-6; do
        for P in Green Green_Jacobi dual; do
            NPIX=$N VOID_STIFFNESS=$K $PY polycrystal_crack_linear.py $P 2>&1 | grep -E "^N=|Error|error" \
                | tee -a logs/stage1_small.log
        done
    done
done

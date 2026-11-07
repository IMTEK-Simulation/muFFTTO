#!/bin/bash
# mesh study, level h = 1/32: the pulse in the periodic channel (length 3), wall-FFT corrected and projected Green
cd "$(dirname "$0")"
export CUDA_PATH=~/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu
for P in ${PRECONDS:-G_wall G}; do
  WAVE=1 PULSE=1 PERIODIC=1 BAR_L=3 NNN=32 P_MAX=12 NB_INC=12 NB_T=48 PRECOND=$P \
    ~/micromamba/envs/muffto-gpu/bin/python pressurized_bar_3D.py > pulse_${P}_n32.log 2>&1
done

#!/bin/bash
# periodic open channel (3D, h = 1/32, GPU, wall-FFT corrected Green): follower uniform, dead load uniform, follower p(z)
cd "$(dirname "$0")"
export CUDA_PATH=~/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu
PY=~/micromamba/envs/muffto-gpu/bin/python
PERIODIC=1 NNN=32 P_MAX=12 NB_INC=12 PRECOND=G_wall LOAD=follower         $PY pressurized_bar_3D.py > channel3d_follower_n32.log 2>&1
PERIODIC=1 NNN=32 P_MAX=12 NB_INC=12 PRECOND=G_wall LOAD=dead             $PY pressurized_bar_3D.py > channel3d_dead_n32.log 2>&1
PERIODIC=1 NNN=32 P_MAX=12 NB_INC=12 PRECOND=G_wall LOAD=follower P_VAR=0.5 $PY pressurized_bar_3D.py > channel3d_pvar_n32.log 2>&1

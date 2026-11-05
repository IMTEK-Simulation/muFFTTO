#!/bin/bash
# after the travelling-wave run: a single pulse (one sine wavelength over the first third) moving along a channel of
# length 3 (h = 1/32, 32 x 32 x 96 voxels), then video + muEye files of both runs
cd "$(dirname "$0")"
export CUDA_PATH=~/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu
PY=~/micromamba/envs/muffto-gpu/bin/python
until grep -q elapsed channel3d_wave_n32.log; do sleep 5; done
WAVE=1 PULSE=1 PERIODIC=1 BAR_L=3 NNN=32 P_MAX=12 NB_INC=12 NB_T=48 N_PERIODS=1 PRECOND=G_wall LOAD=follower \
  $PY pressurized_bar_3D.py > channel3d_pulse_L3_n32.log 2>&1
for R in $(ls -td runs/bar3d_n32_*per*_wave_G_wall_gpu_* | head -1) $(ls -td runs/bar3d_n32_*per_L3*_pulse_G_wall_gpu_* | head -1); do
  $PY bar3d_to_mueye.py $R
  $PY bar3d_video.py $R 6
done

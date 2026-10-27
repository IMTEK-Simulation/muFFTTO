#!/bin/bash
# JCTAM setup, kappa_void = 1e-3, dual preconditioner on larger grids, on the GPU (all at once)
#   nohup ./run_JCTAM_soft3_dual_gpu.sh &      -> logs/JCTAM_N<N>_soft3_dual.log
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu MPLBACKEND=Agg SHOW_PLOTS=0 OMP_NUM_THREADS=2 CUDA_PATH=~/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false
for n in 128 256 512; do
  DEVICE=gpu NPIX=$n SOFT_EXP=3 ~/micromamba/envs/muffto-gpu/bin/python -u example_2D_elasticity_TO_JCTAM.py dual > logs/JCTAM_N${n}_soft3_dual.log 2>&1 &
done
wait
echo "$(date +%H:%M:%S) soft3 dual 128-512 (GPU) done" >> logs/JCTAM_runs.log

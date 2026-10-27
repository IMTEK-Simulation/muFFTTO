#!/bin/bash
# JCTAM setup with kappa_void = 1e-3, grids 16^2-64^2, Green / Green-Jacobi / dual, 9 runs in parallel on the CPU
#   nohup ./run_JCTAM_soft3.sh &      -> logs/JCTAM_N<N>_soft3_<prec>.log
cd "$(dirname "$0")"
for n in 16 32 64; do for p in Green Green_Jacobi dual; do echo "$n $p"; done; done | \
  xargs -P 9 -L 1 bash -c 'JAX_PLATFORMS=cpu MPLBACKEND=Agg OMP_NUM_THREADS=1 SHOW_PLOTS=0 NPIX=$0 SOFT_EXP=3 \
    ~/micromamba/envs/muffto-gpu/bin/python -u example_2D_elasticity_TO_JCTAM.py $1 > logs/JCTAM_N$0_soft3_$1.log 2>&1'
echo "$(date +%H:%M:%S) soft3 16-64 done" >> logs/JCTAM_runs.log

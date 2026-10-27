#!/bin/bash
# JCTAM setup (see example_2D_elasticity_TO_JCTAM.py), grids 16^2-256^2, on the CPU, 12 runs at a time, small grids first:
#   kappa_void = 0:    Green-Jacobi 16-256, Green 16-64 (as in JCTAM: FFT hits the 10 000 iteration limit beyond)
#   kappa_void = 1e-5: Green, Green-Jacobi, dual 16-256
#   nohup ./run_JCTAM_setup.sh &      -> logs/JCTAM_N<N>_soft<e>_<prec>.log
cd "$(dirname "$0")"
{
for n in 16 32 64 128 256; do
  echo "$n 0 Green_Jacobi"; [ $n -le 64 ] && echo "$n 0 Green"
  for p in Green Green_Jacobi dual; do echo "$n 5 $p"; done
done
} | xargs -P 12 -L 1 bash -c 'JAX_PLATFORMS=cpu MPLBACKEND=Agg OMP_NUM_THREADS=1 SHOW_PLOTS=0 NPIX=$0 SOFT_EXP=$1 \
    ~/micromamba/envs/muffto-gpu/bin/python -u example_2D_elasticity_TO_JCTAM.py $2 > logs/JCTAM_N$0_soft$1_$2.log 2>&1'
echo "$(date +%H:%M:%S) all done" >> logs/JCTAM_runs.log

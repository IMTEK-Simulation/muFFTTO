#!/bin/bash
# JCTAM setup, kappa_void = 1e-3: cost estimate of Green / Green-Jacobi / dual on a formed design. Every run starts from
# frame 500 of the dual run on the same grid and does only a few L-BFGS iterations (MAXITER=6).
#   nohup ./run_JCTAM_soft3_from_dual.sh &      -> logs/JCTAM_N<N>_soft3_<prec>_fromdual.log
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu MPLBACKEND=Agg SHOW_PLOTS=0 OMP_NUM_THREADS=2 CUDA_PATH=~/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false
export SOFT_EXP=3 MAXITER=6 INIT_FRAME=500 RUN_SUFFIX=_fromdual_f500
for n in 32 64 128 256 512; do
  device=gpu; [ $n -le 128 ] && device=cpu
  for p in Green Green_Jacobi dual; do
    DEVICE=$device NPIX=$n INIT_FROM=data/example_2D_elasticity_TO_JCTAM/N${n}_soft0.001_residual_dual_eta_0.01_w_20.0_l_bfgs_history.nc \
      ~/micromamba/envs/muffto-gpu/bin/python -u example_2D_elasticity_TO_JCTAM.py $p > logs/JCTAM_N${n}_soft3_${p}_fromdual.log 2>&1 &
  done
done
wait
echo "$(date +%H:%M:%S) soft3 from dual frame 500 done" >> logs/JCTAM_runs.log

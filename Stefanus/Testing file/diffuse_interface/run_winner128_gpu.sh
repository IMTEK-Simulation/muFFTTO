#!/bin/bash
# GPU wall time at 128 px, 2 increments, fast Hessian: Green (plain medium, sharp) vs winner (dual ref 10 + Green-norm
# trust region + graded medium CAP 1e-3, w 2)
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
C="-n 128 -inc 2 --eps_max 0.01 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --incl_contrast 10 --lam_over_mu 100 --third_medium 1e-4 --interp log --device gpu --verbose"
run() { CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg \
          OMP_NUM_THREADS=2 $PY stefanus_diffuse.py $C $2 --tag _gpu128_$1 > exp_data/log_gpu128_$1.txt 2>&1; }
run green "--precond green"
run winner "--diffuse 2 --void_grade 1e-3 --precond dual --ref_lam_over_mu 10 --tr_metric green"

#!/bin/bash
# nonlinear wall-time comparison with the closed-form dual middle operator: 64 px, sharp interfaces (--interp log
# --diffuse 0), simplified material, third medium k = 1e-4, 2 increments; dual + Green-norm trust region vs Green; CPU
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
C="-n 64 -inc 2 --eps_max 0.01 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --incl_contrast 10 --lam_over_mu 100 --third_medium 1e-4 --verbose"
run() { OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu MPLBACKEND=Agg $PY stefanus_diffuse.py $C $2 --tag _fast_$1 > exp_data/log_fast_nonlinear64_$1.txt 2>&1; }
run dual_trgreen "--precond dual --tr_metric green" &
run green "--precond green" &
wait

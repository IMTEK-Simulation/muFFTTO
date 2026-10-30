#!/bin/bash
# wall time at 64 px, CPU, 2 increments, fast Hessian: Green on the plain third medium 1e-4 (sharp) vs the winning set-up
# (dual ref lambda/mu 10 + Green-norm trust region + graded medium CAP 1e-3, w 2)
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
C="-n 64 -inc 2 --eps_max 0.01 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --incl_contrast 10 --lam_over_mu 100 --third_medium 1e-4 --interp log --verbose"
run() { OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu MPLBACKEND=Agg $PY stefanus_diffuse.py $C $2 --tag _winner_$1 > exp_data/log_winner64_$1.txt 2>&1; }
run winner "--diffuse 2 --void_grade 1e-3 --precond dual --ref_lam_over_mu 10 --tr_metric green" &
wait

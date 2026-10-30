#!/bin/bash
# robustness through buckling: 64 px, sharp interfaces, simplified material, third medium 1e-4, 10 increments to
# eps 0.05; Green vs dual (ref lambda/mu 10, Green-norm trust region); CPU, fast Hessian
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
C="-n 64 -inc 10 --eps_max 0.05 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --incl_contrast 10 --lam_over_mu 100 --third_medium 1e-4 --interp log --diffuse 0 --verbose --frames"
run() { OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu MPLBACKEND=Agg $PY stefanus_diffuse.py $C $2 --tag _buckle_$1 > exp_data/log_buckle64_$1.txt 2>&1; }
run green "--precond green" &
run dual_ref10 "--precond dual --ref_lam_over_mu 10 --tr_metric green" &
wait

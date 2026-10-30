#!/bin/bash
# the winning set-up over the full contact path at 64 px (CPU): dual ref lambda/mu 10 + Green-norm trust region + graded
# third medium (1e-4 -> 1e-3 at the walls, w 2) + fast Hessian, to eps 0.2 in 40 increments.
# Reference: ../exp_data/log_contact64_TM1e-4.txt (Green, plain medium, old Hessian: 2.38 M CG, 5.95 h)
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu MPLBACKEND=Agg $PY stefanus_diffuse.py -n 64 -inc 40 --eps_max 0.2 --direction X --cells 1x2 \
  --imperfection 1e-3 --element bilinear_rectangle --sri --incl_contrast 10 --lam_over_mu 100 --third_medium 1e-4 --interp log \
  --diffuse 2 --void_grade 1e-3 --precond dual --ref_lam_over_mu 10 --tr_metric green --frames --verbose \
  --tag _winner_contact > exp_data/log_winner_contact64.txt 2>&1

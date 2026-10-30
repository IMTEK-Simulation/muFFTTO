#!/bin/bash
# dual preconditioner in the Steihaug CG with the trust region in the Green norm: 64 px, simplified material
# (inclusion contrast 10, Lambda/mu 100), third medium k = 1e-4, 2 increments to eps 0.01.
# Reference (Green, Green norm): ../exp_data/log_contact64_TM1e-4.txt, increments 1, 2: TR 5 / 6, CG 31 537 / 39 417
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu MPLBACKEND=Agg $PY stefanus_dual_trnorm.py -n 64 -inc 2 --eps_max 0.01 --direction X \
  --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --incl_contrast 10 --lam_over_mu 100 \
  --third_medium 1e-4 --precond dual --tr_metric green --verbose --trace --tag _dual_trgreen > exp_data/log_dual_trgreen_TM1e-4.txt 2>&1

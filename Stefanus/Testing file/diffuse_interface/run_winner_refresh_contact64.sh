#!/bin/bash
# winner + dual refresh (per load increment) from the eps = 0.08 state of the winner contact run, to eps 0.2 (64 px, CPU);
# compare with the running/finished run without refresh (exp_data/log_winner_contact64.txt)
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu MPLBACKEND=Agg $PY stefanus_diffuse.py -n 64 -inc 40 --eps_max 0.2 --direction X --cells 1x2 \
  --imperfection 1e-3 --element bilinear_rectangle --sri --incl_contrast 10 --lam_over_mu 100 --third_medium 1e-4 --interp log \
  --diffuse 2 --void_grade 1e-3 --precond dual --ref_lam_over_mu 10 --tr_metric green --dual_refresh --frames --verbose \
  --restart_frame exp_data/frames_TR_X_64x110_cells1x2_TM0.0001_sri_winner_contact/inc0016.npz \
  --tag _winner_refresh > exp_data/log_winner_refresh_contact64.txt 2>&1

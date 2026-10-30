#!/bin/bash
# full contact path at 64 px (CPU) with a gentler grade that should keep the contact physics: w 1 (physical width of
# 2 px at 128), CAP 3e-4, dual ref 10 + Green-norm trust region + refresh; default CG cap (as Green's run)
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
OMP_NUM_THREADS=6 JAX_PLATFORMS=cpu MPLBACKEND=Agg $PY stefanus_diffuse.py -n 64 -inc 40 --eps_max 0.2 --direction X --cells 1x2 \
  --imperfection 1e-3 --element bilinear_rectangle --sri --incl_contrast 10 --lam_over_mu 100 --third_medium 1e-4 --interp log \
  --diffuse 1 --void_grade 3e-4 --precond dual --ref_lam_over_mu 10 --tr_metric green --dual_refresh --frames --verbose \
  --tag _gentle_contact > exp_data/log_gentle_contact64.txt 2>&1

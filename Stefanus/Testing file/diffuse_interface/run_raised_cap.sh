#!/bin/bash
# one contact-phase increment (eps 0.15 -> 0.155, from the winner frame) with the Steihaug cap raised to 60 000 (exact
# 1e-6 solves): Green vs dual ref 10 vs dual ref 10 + refresh; GPU, 64 px
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
F=exp_data/frames_TR_X_64x110_cells1x2_TM0.0001_sri_winner_contact/inc0030.npz
C="-n 64 -inc 31 --eps_max 0.155 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --incl_contrast 10 --lam_over_mu 100 --third_medium 1e-4 --interp log --diffuse 2 --void_grade 1e-3 --device gpu --restart_frame $F --inner_maxiter 60000"
for a in "--precond green" "--precond dual --ref_lam_over_mu 10 --tr_metric green" "--precond dual --ref_lam_over_mu 10 --tr_metric green --dual_refresh"; do echo "$a"; done \
  | xargs -P 3 -I{} sh -c "CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg \
      OMP_NUM_THREADS=2 timeout 3h $PY stefanus_diffuse.py $C {} 2>&1 | grep '^inc\|TR its, \|Error\|Traceback' | sed 's/^/[{}] /'" \
  > exp_data/log_raised_cap.txt

#!/bin/bash
# augmented-Lagrangian test (stefanus_AL.py): 64 px, true holes, simplified material (inclusion contrast 10,
# Lambda/mu 100), 2 increments to eps 0.01, penalty r = 10 mu_m. Reference without AL: ../exp_data/log_contact64_holes.txt
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
mkdir -p exp_data
if [ -z "$1" ]; then
OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu MPLBACKEND=Agg $PY stefanus_AL.py -n 64 -inc 2 --eps_max 0.01 --direction X \
  --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --precond green --incl_contrast 10 --lam_over_mu 100 \
  --al_r 10 --verbose --tag _AL_r10 > exp_data/log_AL_r10_holes.txt 2>&1
fi

# same with the voids filled by a third medium k = 1e-4 (nu = 0.3). Reference without AL:
# ../exp_data/log_contact64_TM1e-4.txt (increments 1, 2: 31 537 and 39 417 CG)
if [ "$1" == "tm" ]; then
OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu MPLBACKEND=Agg $PY stefanus_AL.py -n 64 -inc 2 --eps_max 0.01 --direction X \
  --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --precond green --incl_contrast 10 --lam_over_mu 100 \
  --third_medium 1e-4 --al_r 10 --verbose --tag _AL_r10 > exp_data/log_AL_r10_TM1e-4.txt 2>&1
fi

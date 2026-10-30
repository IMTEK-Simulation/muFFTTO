#!/bin/bash
# mixed displacement-pressure linear test (stefanus_mixed.py --mixed_test) at F = I, 64 px, simplified material
# (inclusion contrast 10, Lambda/mu 100), third medium k = 1e-4 (nu = 0.3); argument: third medium k (default 1e-4)
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
TM=${1:-1e-4}
OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu MPLBACKEND=Agg $PY stefanus_mixed.py -n 64 -inc 1 --eps_max 0.0005 --direction X \
  --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --incl_contrast 10 --lam_over_mu 100 \
  --third_medium $TM --mixed_test > exp_data/log_mixed_test_TM$TM.txt 2>&1

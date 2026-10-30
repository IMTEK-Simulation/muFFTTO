#!/bin/bash
# nonlinear test with 2 px diffuse interfaces at 128 px (GPU): simplified material (inclusion contrast 10, Lambda/mu 100),
# third medium k = 1e-4, 2 increments to eps 0.01; (a) dual in CG + trust region in the Green norm, (b) Green
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
COMMON="-n 128 -inc 2 --eps_max 0.01 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --incl_contrast 10 --lam_over_mu 100 --third_medium 1e-4 --diffuse 2 --device gpu --verbose"
run() {  # name, extra args
  CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg \
    OMP_NUM_THREADS=2 $PY stefanus_diffuse.py $COMMON $2 --tag _w2_$1 > exp_data/log_nonlinear_w2_$1.txt 2>&1
}
run dual_trgreen "--precond dual --tr_metric green" &
run green "--precond green" &
wait

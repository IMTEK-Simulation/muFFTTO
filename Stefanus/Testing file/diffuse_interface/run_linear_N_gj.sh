#!/bin/bash
# Green-Jacobi on the linearly N-interpolated interfaces (as run_linear_N.sh part A): 128 px, GPU, linear test at F = I,
# widths 0, 1, 2 px, simplified material, third medium k = 1e-4
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
ENV="CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg OMP_NUM_THREADS=2"
MAT="-n 128 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --incl_contrast 10 --lam_over_mu 100 --third_medium 1e-4 --device gpu"
for W in 0 1 2; do echo "-inc 1 --eps_max 0.0005 --linear_test --interp linear_N --diffuse $W --precond gj"; done \
  | xargs -P 3 -I{} sh -c "env $ENV timeout 3h $PY stefanus_diffuse.py $MAT {} 2>&1 | grep 'CG iterations\|Error' \
    | sed 's/^/{}  /'" > exp_data/log_linear_N_gj.txt

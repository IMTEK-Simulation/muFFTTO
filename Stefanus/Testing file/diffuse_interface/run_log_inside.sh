#!/bin/bash
# one-sided log ramp (stefanus_diffuse.py --interp log_inside): the matrix stays sharp, only the void / inclusion interior
# ramps. 128 px, GPU, simplified material, third medium k = 1e-4.
#   (A) linear test at F = I: widths 1, 2 px, Green and dual
#   (B) one load increment (eps 0.005), dual in CG + Green-norm trust region: sharp, log_inside w 1, w 2 -> stress, cost
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
ENV="CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg OMP_NUM_THREADS=2"
MAT="-n 128 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --incl_contrast 10 --lam_over_mu 100 --third_medium 1e-4 --device gpu"
{
for W in 1 2; do for P in green dual; do echo "-inc 1 --eps_max 0.0005 --linear_test --interp log_inside --diffuse $W --precond $P"; done; done
} | xargs -P 4 -I{} sh -c "env $ENV timeout 2h $PY stefanus_diffuse.py $MAT {} 2>&1 | grep 'CG iterations\|Error' \
    | sed 's/^/{}  /'" > exp_data/log_log_inside_lineartest.txt
{
echo "--interp log --diffuse 0 --tag _sharp"
echo "--interp log_inside --diffuse 1 --tag _login_w1"
echo "--interp log_inside --diffuse 2 --tag _login_w2"
} | xargs -P 3 -I{} sh -c "env $ENV timeout 3h $PY stefanus_diffuse.py $MAT -inc 1 --eps_max 0.005 --precond dual \
    --tr_metric green {} 2>&1 | grep '^inc\|TR its, \|Error\|failed' | sed 's/^/{}  /'" > exp_data/log_log_inside_nonlinear.txt

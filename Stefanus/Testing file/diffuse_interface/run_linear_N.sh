#!/bin/bash
# linear interpolation of the moduli through the N operator (stefanus_diffuse.py --interp linear_N), 128 px, GPU,
# simplified material, third medium k = 1e-4:
#   (A) linear test at F = I: widths 0, 1, 2 px, Green and dual
#   (B) one load increment (eps 0.005) with dual in CG + Green-norm trust region: sharp, linear_N w 0, linear_N w 1
#       -> stress against the sharp interface, and the cost
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
ENV="CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg OMP_NUM_THREADS=2"
MAT="-n 128 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --incl_contrast 10 --lam_over_mu 100 --third_medium 1e-4 --device gpu"
{
for W in 0 1 2; do for P in green dual; do echo "-inc 1 --eps_max 0.0005 --linear_test --interp linear_N --diffuse $W --precond $P"; done; done
} | xargs -P 3 -I{} sh -c "env $ENV timeout 2h $PY stefanus_diffuse.py $MAT {} 2>&1 | grep 'CG iterations\|Error' \
    | sed 's/^/{}  /'" > exp_data/log_linear_N_lineartest.txt
{
echo "--interp log --diffuse 0 --tag _sharp"
echo "--interp linear_N --diffuse 0 --tag _linN_w0"
echo "--interp linear_N --diffuse 1 --tag _linN_w1"
} | xargs -P 3 -I{} sh -c "env $ENV timeout 3h $PY stefanus_diffuse.py $MAT -inc 1 --eps_max 0.005 --precond dual \
    --tr_metric green {} 2>&1 | grep '^inc\|TR its, \|Error\|failed' | sed 's/^/{}  /'" > exp_data/log_linear_N_nonlinear.txt

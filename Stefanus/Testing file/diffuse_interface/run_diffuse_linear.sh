#!/bin/bash
# diffuse interfaces (stefanus_diffuse.py --diffuse w): linear test at F = I, 64 px, simplified material (inclusion
# contrast 10, Lambda/mu 100), third medium k = 1e-4; widths 0 (sharp), 1, 2 px; Green, Green-Jacobi, dual
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
COMMON="-n 64 -inc 1 --eps_max 0.0005 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --linear_test --incl_contrast 10 --lam_over_mu 100 --third_medium 1e-4"
{
for W in 0 1 2; do for P in green gj dual; do echo "--diffuse $W --precond $P"; done; done
} | xargs -P 9 -I{} sh -c "OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu MPLBACKEND=Agg timeout 2h $PY stefanus_diffuse.py $COMMON {} 2>&1 \
    | grep 'CG iterations\|diffuse interfaces\|Error' | tr '\n' ' ' | sed 's/\$/\n/'; echo" > exp_data/log_diffuse_linear.txt

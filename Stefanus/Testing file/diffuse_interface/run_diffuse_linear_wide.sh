#!/bin/bash
# wider diffuse interfaces at 128 px (see run_diffuse_linear_large.sh): widths 4 and 8 px, Green, Green-Jacobi, dual;
# GPU, linear test at F = I, simplified material, third medium k = 1e-4. NB: the thinnest ligament spans ~6 px at 128,
# so these bands also soften it (see figures/material_diffuse_N128_wide.png)
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
COMMON="-n 128 -inc 1 --eps_max 0.0005 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --linear_test --incl_contrast 10 --lam_over_mu 100 --third_medium 1e-4 --device gpu"
{
for W in 4 8; do for P in green gj dual; do echo "--diffuse $W --precond $P"; done; done
} | xargs -P 3 -I{} sh -c "CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MPLBACKEND=Agg OMP_NUM_THREADS=2 timeout 3h $PY stefanus_diffuse.py $COMMON {} 2>&1 \
    | grep 'CG iterations\|Segmentation\|Error' | sed 's/^/-n 128 {}  /'" > exp_data/log_diffuse_linear_wide.txt

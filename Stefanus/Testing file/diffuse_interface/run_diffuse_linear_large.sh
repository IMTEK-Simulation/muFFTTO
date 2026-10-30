#!/bin/bash
# diffuse interfaces on finer grids (the ligament spans ~6 px at 128, ~12 px at 256, so a 1-2 px band does not eat it):
# linear test at F = I, simplified material, third medium k = 1e-4; widths 0, 1, 2 px; Green, Green-Jacobi, dual; GPU
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
COMMON="-inc 1 --eps_max 0.0005 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --linear_test --incl_contrast 10 --lam_over_mu 100 --third_medium 1e-4 --device gpu"
{
for N in 128 256; do for W in 0 1 2; do for P in green gj dual; do echo "-n $N --diffuse $W --precond $P"; done; done; done
} | xargs -P 3 -I{} sh -c "CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MPLBACKEND=Agg OMP_NUM_THREADS=2 timeout 3h $PY stefanus_diffuse.py $COMMON {} 2>&1 \
    | grep 'CG iterations\|Segmentation\|Error' | sed 's/^/{}  /'" > exp_data/log_diffuse_linear_large.txt

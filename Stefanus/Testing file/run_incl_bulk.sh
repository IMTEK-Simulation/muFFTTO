#!/bin/bash
# inclusion with bulk modulus = shear modulus (lambda / mu = 1/3, nu = 0.125) instead of lambda / mu = 100:
# (1) response: true holes, 128 px, GPU, to eps 0.05 (compare log_simplify128_c10_L100.txt)
# (2) cost: linear test at F = I, 64 px, third medium 1e-3 / 1e-4, Green and dual (CPU)
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
MAT="--incl_contrast 10 --lam_over_mu 100 --incl_lam_over_mu 0.3333333"
systemd-run --user --scope -p MemoryMax=10G -q env CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux \
  XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg OMP_NUM_THREADS=2 \
  $PY stefanus_2D_finite_strain_NeoHookean_TR.py -n 128 -inc 10 --eps_max 0.05 --direction X --cells 1x2 \
  --imperfection 1e-3 --element bilinear_rectangle --sri --precond green --device gpu --frames $MAT --tag _c10_L100_IK1 \
  > exp_data/log_simplify128_c10_L100_IK1.txt 2>&1 &
LIN="-n 64 -inc 1 --eps_max 0.0005 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --linear_test $MAT"
{
for P in green dual; do for TM in 1e-3 1e-4; do echo "--precond $P --third_medium $TM"; done; done
echo "--precond green"
} | xargs -P 5 -I{} sh -c "systemd-run --user --scope -p MemoryMax=6G -q env OMP_NUM_THREADS=1 MPLBACKEND=Agg \
    JAX_PLATFORMS=cpu timeout 1h $PY stefanus_2D_finite_strain_NeoHookean_TR.py $LIN {} 2>&1 | grep 'CG iterations\|Error'" \
  > exp_data/log_linear_incl_bulk.txt
wait

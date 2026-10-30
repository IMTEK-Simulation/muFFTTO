#!/bin/bash
# linear test at F = I (see run_linear_test.sh) with the simplified material (inclusion contrast 10, Lambda/mu 100):
# third medium k = 1e-3 / 1e-4 / 1e-5 (nu = 0.3), Green vs dual, 64 px
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
COMMON="-n 64 -inc 1 --eps_max 0.0005 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --linear_test --incl_contrast 10 --lam_over_mu 100"
{
for P in green dual; do
  for TM in 1e-3 1e-4 1e-5; do echo "--precond $P --third_medium $TM"; done
done
echo "--precond green"
} | xargs -P 7 -I{} sh -c "systemd-run --user --scope -p MemoryMax=6G -q env OMP_NUM_THREADS=1 MPLBACKEND=Agg \
    JAX_PLATFORMS=cpu timeout 2h $PY stefanus_2D_finite_strain_NeoHookean_TR.py $COMMON {} 2>&1 | grep 'CG iterations\|Error'"

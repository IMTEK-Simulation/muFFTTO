#!/bin/bash
# linear test at F = I (see run_linear_test.sh): third medium (a) lambda 1e-3, mu 1e-6 and (b) mu 1e-3, lambda 1e-6;
# inclusion contrast 1000 and 100; Green vs dual; 32 px
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
COMMON="-n 32 -inc 1 --eps_max 0.0005 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --linear_test"
{
for P in green dual; do
  for C in 1000 100; do
    echo "--precond $P --incl_contrast $C --third_medium 1e-6 --tm_lam_over_mu 1000"
    echo "--precond $P --incl_contrast $C --third_medium 1e-3 --tm_lam_over_mu 1e-3"
  done
done
} | xargs -P 8 -I{} sh -c "systemd-run --user --scope -p MemoryMax=6G -q env OMP_NUM_THREADS=1 MPLBACKEND=Agg \
    JAX_PLATFORMS=cpu timeout 2h $PY stefanus_2D_finite_strain_NeoHookean_TR.py $COMMON {} 2>&1 | grep 'CG iterations\|Error'"

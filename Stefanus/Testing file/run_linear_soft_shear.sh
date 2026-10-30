#!/bin/bash
# linear test at F = I (see run_linear_test.sh): third medium with lambda as before (1.5e-3 / 1.5e-5 mu_m) and
# lambda / mu = 1000 (as the matrix), i.e. mu = 1.5e-6 / 1.5e-8 mu_m; inclusion contrast 1000 and 100; Green vs dual
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
COMMON="-inc 1 --eps_max 0.0005 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --linear_test"
{
for N in 32 64; do
  for P in green dual; do
    for C in 1000 100; do
      for MU in 1.5e-6 1.5e-8; do
        echo "-n $N --precond $P --incl_contrast $C --third_medium $MU --tm_lam_over_mu 1000"
      done
    done
  done
done
} | xargs -P 14 -I{} sh -c "systemd-run --user --scope -p MemoryMax=6G -q env OMP_NUM_THREADS=1 MPLBACKEND=Agg \
    JAX_PLATFORMS=cpu timeout 2h $PY stefanus_2D_finite_strain_NeoHookean_TR.py $COMMON {} 2>&1 | grep 'CG iterations\|Error'"

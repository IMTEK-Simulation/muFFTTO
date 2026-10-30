#!/bin/bash
# linear test at F = I (see run_linear_test.sh): third medium lambda / mu 1000 (as the matrix) vs 1.5, inclusion
# contrast 100 vs 1000, Green vs dual; at most 14 jobs at a time
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
COMMON="-inc 1 --eps_max 0.0005 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --linear_test"
{
for N in 32 64; do
  echo "-n $N --precond green --incl_contrast 100"
  for P in green dual; do
    for C in 1000 100; do
      for TM in 1e-3 1e-5; do
        for R in 1000 1.5; do
          echo "-n $N --precond $P --incl_contrast $C --third_medium $TM --tm_lam_over_mu $R"
        done
      done
    done
  done
done
} | xargs -P 14 -I{} sh -c "systemd-run --user --scope -p MemoryMax=6G -q env OMP_NUM_THREADS=1 MPLBACKEND=Agg \
    JAX_PLATFORMS=cpu timeout 2h $PY stefanus_2D_finite_strain_NeoHookean_TR.py $COMMON {} 2>&1 | grep 'CG iterations\|Error'"

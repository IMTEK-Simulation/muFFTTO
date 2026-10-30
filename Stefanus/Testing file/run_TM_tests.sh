#!/bin/bash
# third-medium contact test, 64 x 110 (1x2 cells), SRI Q1, X compression to eps 0.015 in 3 increments (buckling at
# ~0.016): CG per trust-region step with the fixed 1e-6 energy tolerance and adaptive delay (min 10)
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
COMMON="-n 64 -inc 3 --eps_max 0.015 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --verbose"
run() {  # name, extra args
  systemd-run --user --scope -p MemoryMax=8G -q env OMP_NUM_THREADS=2 MPLBACKEND=Agg JAX_PLATFORMS=cpu \
    timeout 4h $PY stefanus_2D_finite_strain_NeoHookean_TR.py $COMMON $2 --tag _tol1e-6_$1 > exp_data/log_TMtest_$1.txt 2>&1
}
run holes_green "--precond green" &
run TM1e-5_green "--third_medium 1e-5 --precond green" &
run TM1e-5_gj "--third_medium 1e-5 --precond gj" &
run TM1e-4_green "--third_medium 1e-4 --precond green" &
run TM1e-4_gj "--third_medium 1e-4 --precond gj" &
wait

#!/bin/bash
# third-medium test with a 10x smaller load step: one increment of eps 5e-4 (else as run_TM_tests.sh)
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
COMMON="-n 64 -inc 1 --eps_max 0.0005 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --verbose"
run() {  # name, extra args
  systemd-run --user --scope -p MemoryMax=8G -q env OMP_NUM_THREADS=2 MPLBACKEND=Agg JAX_PLATFORMS=cpu \
    timeout 4h $PY stefanus_2D_finite_strain_NeoHookean_TR.py $COMMON $2 --tag _tol1e-6_tiny_$1 > exp_data/log_TMtiny_$1.txt 2>&1
}
run holes_green "--precond green" &
run TM1e-5_green "--third_medium 1e-5 --precond green" &
run TM1e-4_green "--third_medium 1e-4 --precond green" &
wait

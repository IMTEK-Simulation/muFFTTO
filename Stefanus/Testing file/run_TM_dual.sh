#!/bin/bash
# third medium 1e-3: Green vs dual (G K_dual G), small grids, one load increment eps 5e-4, traced
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
COMMON="-inc 1 --eps_max 0.0005 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --verbose --trace --third_medium 1e-3"
run() {  # name, extra args
  systemd-run --user --scope -p MemoryMax=8G -q env OMP_NUM_THREADS=2 MPLBACKEND=Agg JAX_PLATFORMS=cpu \
    timeout 3h $PY stefanus_2D_finite_strain_NeoHookean_TR.py $COMMON $2 --tag _tol1e-6_tiny_$1 > exp_data/log_TMdual_$1.txt 2>&1
}
for N in 32 64; do
  run N${N}_TM1e-3_green "-n $N --precond green" &
  run N${N}_TM1e-3_dual "-n $N --precond dual" &
done
wait

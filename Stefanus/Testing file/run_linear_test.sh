#!/bin/bash
# H(F = I) s = -g (g of the first increment eps 5e-4 at the initial state) by plain PCG, energy stop 1e-6 with
# adaptive delay: CG iterations of Green / Green-Jacobi / dual for true holes and third medium 1e-3, 1e-5
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
COMMON="-inc 1 --eps_max 0.0005 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --linear_test"
run() {
  systemd-run --user --scope -p MemoryMax=6G -q env OMP_NUM_THREADS=1 MPLBACKEND=Agg JAX_PLATFORMS=cpu \
    timeout 2h $PY stefanus_2D_finite_strain_NeoHookean_TR.py $COMMON $1 2>&1 | grep "linear test\|Error\|assert"
}
for N in 32 64; do
  for P in green gj; do run "-n $N --precond $P" & done
  for TM in 1e-3 1e-5; do
    for P in green gj dual; do run "-n $N --precond $P --third_medium $TM" & done
  done
done
wait

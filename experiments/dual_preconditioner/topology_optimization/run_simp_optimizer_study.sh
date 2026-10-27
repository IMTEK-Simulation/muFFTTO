#!/bin/bash
# Optimizer study on the SIMP problem (example_2D_elasticity_TO_simp_mma.py), 64^2, void 1e-5, Green-Jacobi:
# L-BFGS-B, GCMMA and the plain MMA, each from the start designs INIT_SEED = 0..4, all other settings identical
# (default schedule, filter, stage stop, CG). 6 runs at a time, 2 threads each. Logs: logs/simp_<opt>_N64_GJ_seed<s>.log
cd "$(dirname "$0")"
PY=/home/martin/micromamba/envs/muffto-gpu/bin/python
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2
for seed in 0 1 2 3 4; do
  for opt in lbfgs gcmma mma; do
    log=logs/simp_${opt}_N64_GJ_seed${seed}.log
    OPTIMIZER=$opt INIT_SEED=$seed $PY -u example_2D_elasticity_TO_simp_mma.py Green_Jacobi > $log 2>&1 &
    while [ "$(jobs -rp | wc -l)" -ge 6 ]; do sleep 5; done
  done
done
wait
echo done

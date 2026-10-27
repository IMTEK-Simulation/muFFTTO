#!/bin/bash
# Optimizer study on the phase-field formulation (example_2D_elasticity_TO_phase_field_optimizers.py: the objective
# of example_2D_elasticity_TO.py), 64^2, void 1e-5, eta 1/32, Green-Jacobi: L-BFGS-B, GCMMA and the plain MMA, each
# from the start designs INIT_SEED = 0..4 of the SIMP study, all other settings identical.
# 6 runs at a time, 2 threads each. Logs: logs/pf_<opt>_N64_GJ_seed<s>.log
cd "$(dirname "$0")"
PY=/home/martin/micromamba/envs/muffto-gpu/bin/python
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 NPIX=64 SOFT_EXP=5 ETA=0.03125
for seed in 0 1 2 3 4; do
  for opt in lbfgs gcmma mma; do
    log=logs/pf_${opt}_N64_GJ_seed${seed}.log
    OPTIMIZER=$opt INIT_SEED=$seed $PY -u example_2D_elasticity_TO_phase_field_optimizers.py Green_Jacobi > $log 2>&1 &
    while [ "$(jobs -rp | wc -l)" -ge 6 ]; do sleep 5; done
  done
done
wait
echo done

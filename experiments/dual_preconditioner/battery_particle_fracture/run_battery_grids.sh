#!/bin/bash
# Grid study 32^2, 64^2, 128^2 (l = 1/16 fixed: l/h = 2, 4, 8) for one elastic preconditioner per process:
#   ./run_battery_grids.sh Green & ./run_battery_grids.sh Green_Jacobi & ./run_battery_grids.sh dual &
# MATRIX_STIFFNESS (env, default 1e-2) is passed through; T_MAX=1.0 (cracking is complete by t ~ 0.8, the CV tail
# afterwards only fills the core). Diffusion preconditioner comparison (shadow solves) in the dual runs only.
cd "$(dirname "$0")"
P=$1
MS=${MATRIX_STIFFNESS:-1e-2}
SHADOW=0; [ "$P" = dual ] && SHADOW=1
for N in ${GRIDS:-32 64 128}; do
  systemd-run --user --scope -p MemoryMax=4G -q env MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
    NPIX=$N MATRIX_STIFFNESS=$MS T_MAX=${T_MAX:-1.0} DIFF_SHADOW=$SHADOW \
    ~/micromamba/envs/muffto-gpu/bin/python -u example_2D_battery_particle_fracture_AT2.py $P \
    > logs/battery_N${N}_${P}_ms${MS}.log 2>&1
done

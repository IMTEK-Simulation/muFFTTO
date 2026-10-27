#!/bin/bash
# AT2 polydisperse pores, 256^2, the three elastic preconditioners on the CPU (4 MPI ranks), one after another for
# clean timings; rerun with the current reference media (the old-C_ref runs are in */old_Cref_relic/)
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu MPLBACKEND=Agg OMP_NUM_THREADS=1
PY=~/micromamba/envs/muffto-gpu/bin/python
for p in dual Green_Jacobi Green; do
    MICROSTRUCTURE=polydisperse_pores NPIX=256 LENGTH_SCALE=0.03125 \
        ~/micromamba/envs/muffto-gpu/bin/mpirun -n 4 --bind-to none $PY -u example_2D_phase_field_fracture_AT2.py $p \
        > logs/AT2_N256_${p}_polydisperse_pores_l0.03125.log 2>&1
done

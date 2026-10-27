#!/bin/bash
# fixed-l AT2 series on the polydisperse pore microstructure, 32^2-256^2 on the CPU (serial up to 64^2, 4 MPI ranks above);
# the three elastic preconditioners in parallel, the grids one after another (GRIDS="128 256" ./run_... for a subset)
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu MPLBACKEND=Agg OMP_NUM_THREADS=1
for p in dual Green_Jacobi Green; do
    (for n in ${GRIDS:-32 64 128 256}; do
        mpi=$([ $n -gt 64 ] && echo "$HOME/micromamba/envs/muffto-gpu/bin/mpirun -n 4 --bind-to none")
        MICROSTRUCTURE=polydisperse_pores NPIX=$n LENGTH_SCALE=0.03125 $mpi ~/micromamba/envs/muffto-gpu/bin/python -u \
            example_2D_phase_field_fracture_AT2.py $p > logs/AT2_N${n}_${p}_polydisperse_pores_l0.03125.log 2>&1
    done) &
done
wait

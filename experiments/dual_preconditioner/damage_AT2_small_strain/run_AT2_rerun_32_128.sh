#!/bin/bash
# AT2 reruns, 32^2-128^2, both series (one pore = paper setup, polydisperse pores), the three elastic preconditioners;
# one run at a time for clean timings: serial up to 64^2, 4 MPI ranks on 128^2
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu MPLBACKEND=Agg OMP_NUM_THREADS=1
PY=~/micromamba/envs/muffto-gpu/bin/python
for micro in circle_inclusion polydisperse_pores; do
    tag=$([ $micro = circle_inclusion ] && echo "" || echo "_$micro")
    for n in 32 64 128; do
        mpi=$([ $n -gt 64 ] && echo "$HOME/micromamba/envs/muffto-gpu/bin/mpirun -n 4 --bind-to none")
        for p in dual Green_Jacobi Green; do
            MICROSTRUCTURE=$micro NPIX=$n LENGTH_SCALE=0.03125 $mpi $PY -u example_2D_phase_field_fracture_AT2.py $p \
                > logs/AT2_N${n}_${p}${tag}_l0.03125.log 2>&1
        done
    done
done

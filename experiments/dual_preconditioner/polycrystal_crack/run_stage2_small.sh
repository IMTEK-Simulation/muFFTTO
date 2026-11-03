#!/bin/bash
# Stage 2, small scale: AT2 crack growth from the edge pre-crack through the polycrystal (POLYCRYSTAL=1),
# elastic preconditioners Green / Green_Jacobi / dual run in parallel (CPU, serial, single thread each).
# l = 0.03125 fixed (l/h = 2 at 64^2, 4 at 128^2), Gc = 1e-5 (units E_min = 1 GPa), crack/void stiffness 1e-4.
# usage:  ./run_stage2_small.sh 64        (one grid per call)
cd "$(dirname "$0")"
N=${1:-64}
PY=~/micromamba/envs/muffto-gpu/bin/python
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
mkdir -p logs
for P in Green Green_Jacobi dual; do
    POLYCRYSTAL=1 NPIX=$N PORE_CONTRAST=1e-4 LENGTH_SCALE=0.03125 GC=1e-5 LOAD_MAX=6e-3 LOAD_STEPS=60 \
        $PY -u example_2D_phase_field_fracture_AT2.py $P > logs/AT2_poly_N${N}_${P}.log 2>&1 &
done
wait
grep -h "Elapsed" logs/AT2_poly_N${N}_*.log

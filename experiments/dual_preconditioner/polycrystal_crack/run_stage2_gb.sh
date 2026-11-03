#!/bin/bash
# Stage 2 with weak grain boundaries: Gc x GC_GB on the 2-pixel grain-boundary band, l = 0.015625 (l/h = 2 at 128^2,
# about a quarter of a grain), otherwise as run_stage2_small.sh. Preconditioners in parallel (CPU, serial each).
# usage:  ./run_stage2_gb.sh 128 0.3
cd "$(dirname "$0")"
N=${1:-128}
GB=${2:-0.3}
PY=~/micromamba/envs/muffto-gpu/bin/python
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
mkdir -p logs
for P in Green Green_Jacobi dual; do
    POLYCRYSTAL=1 GC_GB=$GB NPIX=$N PORE_CONTRAST=1e-4 LENGTH_SCALE=0.015625 GC=1e-5 LOAD_MAX=6e-3 LOAD_STEPS=60 \
        $PY -u example_2D_phase_field_fracture_AT2.py $P > logs/AT2_poly_gb${GB}_N${N}_${P}.log 2>&1 &
done
wait
grep -h "Elapsed" logs/AT2_poly_gb${GB}_N${N}_*.log

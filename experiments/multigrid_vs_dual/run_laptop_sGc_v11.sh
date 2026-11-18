#!/bin/bash
# Laptop 128^2, fixed-strength series _sGc (Gc = 0.016 l): tuned MG = V(1,1), 3 warm power iterations, d_min 10 and 0
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu MPLBACKEND=Agg OMP_NUM_THREADS=1 PYTHONWARNINGS="ignore::RuntimeWarning"
PY=~/micromamba/envs/muffto-gpu/bin/python
N=128
for d in 10 0; do
    suf=_lh2_sGc_v11$([ $d = 0 ] && echo _d0)
    env MICROSTRUCTURE=polydisperse_pores NPIX=$N LENGTH_SCALE=$($PY -c "print(2.0 / $N)") GC=$($PY -c "print(0.032 / $N)") \
        RUN_SUFFIX=$suf CG_DELAY=$d MG_DEGREE=1 MG_POWER=3 \
        $PY -u example_2D_phase_field_fracture_AT2.py MG > logs/AT2_N${N}_MG_polydisperse_pores$suf.log 2>&1 &
done
wait

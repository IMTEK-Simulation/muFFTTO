#!/bin/bash
# Laptop 64^2: cheaper MG cycle V(1,1) (MG_DEGREE=1) with 3 warm power iterations, d_min 10 and 0 (CPU, one thread each)
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu MPLBACKEND=Agg OMP_NUM_THREADS=1 PYTHONWARNINGS="ignore::RuntimeWarning"
PY=~/micromamba/envs/muffto-gpu/bin/python
for d in 10 0; do
    suf=_lh2_v11_pow3$([ $d = 0 ] && echo _d0)
    env MICROSTRUCTURE=polydisperse_pores NPIX=64 LENGTH_SCALE=0.03125 RUN_SUFFIX=$suf MG_DEGREE=1 MG_POWER=3 CG_DELAY=$d \
        $PY -u example_2D_phase_field_fracture_AT2.py MG > logs/AT2_N64_MG_polydisperse_pores$suf.log 2>&1 &
done
wait

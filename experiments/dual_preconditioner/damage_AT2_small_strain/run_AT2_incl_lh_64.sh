#!/bin/bash
# 64^2 small test (2026-10-31): (a) stiff tough inclusions (10x matrix) instead of pores, (c) practical l/h = 2, 4
# (l = 1/32, 1/16); the pores l/h = 2 baseline is the existing AT2_N64_*_polydisperse_pores_l0.03125 series.
# All 9 runs serial and in parallel.
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu MPLBACKEND=Agg OMP_NUM_THREADS=1
PY=~/micromamba/envs/muffto-gpu/bin/python
for p in dual Green_Jacobi Green; do
    for case in "INCL=;L=0.0625" "INCL=10;L=0.03125" "INCL=10;L=0.0625"; do
        eval "$case"
        tag=$([ -n "$INCL" ] && echo "_incl$INCL")_l$L
        env MICROSTRUCTURE=polydisperse_pores NPIX=64 LENGTH_SCALE=$L ${INCL:+INCLUSION_STIFFNESS=$INCL} \
            $PY -u example_2D_phase_field_fracture_AT2.py $p > logs/AT2_N64_${p}_polydisperse_pores${tag}.log 2>&1 &
    done
done
wait

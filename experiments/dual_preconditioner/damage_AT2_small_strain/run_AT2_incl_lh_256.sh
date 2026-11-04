#!/bin/bash
# 256^2 (2026-10-31): pores and stiff tough inclusions (10x), practical l/h = 2, 4 (l = 1/128, 1/64), three elastic
# preconditioners, all 12 runs serial and in parallel. RUN_SUFFIX=_lh marks the fixed-l/h series
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu MPLBACKEND=Agg OMP_NUM_THREADS=1
PY=~/micromamba/envs/muffto-gpu/bin/python
for p in dual Green_Jacobi Green; do
    for case in "INCL=;L=0.0078125" "INCL=;L=0.015625" "INCL=10;L=0.0078125" "INCL=10;L=0.015625"; do
        eval "$case"
        tag=$([ -n "$INCL" ] && echo "_incl$INCL")_l${L}_lh
        env MICROSTRUCTURE=polydisperse_pores NPIX=256 LENGTH_SCALE=$L RUN_SUFFIX=_lh ${INCL:+INCLUSION_STIFFNESS=$INCL} \
            $PY -u example_2D_phase_field_fracture_AT2.py $p > logs/AT2_N256_${p}_polydisperse_pores${tag}.log 2>&1 &
    done
done
wait

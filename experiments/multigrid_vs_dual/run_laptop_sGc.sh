#!/bin/bash
# Laptop 128^2, fixed-strength series _sGc: Gc = 0.016 l (l = 2h), MG and dual, d_min 10 and 0 (CPU, one thread each).
# (64^2 of this series = the standard runs: Gc 5e-4 = 0.016 / 32.)
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu MPLBACKEND=Agg OMP_NUM_THREADS=1 PYTHONWARNINGS="ignore::RuntimeWarning"
PY=~/micromamba/envs/muffto-gpu/bin/python
N=128
for d in 10 0; do
    for P in MG dual; do
        suf=_lh2_sGc$([ $d = 0 ] && echo _d0)
        env MICROSTRUCTURE=polydisperse_pores NPIX=$N LENGTH_SCALE=$($PY -c "print(2.0 / $N)") GC=$($PY -c "print(0.032 / $N)") \
            RUN_SUFFIX=$suf CG_DELAY=$d \
            $PY -u example_2D_phase_field_fracture_AT2.py $P > logs/AT2_N${N}_${P}_polydisperse_pores$suf.log 2>&1 &
    done
done
wait

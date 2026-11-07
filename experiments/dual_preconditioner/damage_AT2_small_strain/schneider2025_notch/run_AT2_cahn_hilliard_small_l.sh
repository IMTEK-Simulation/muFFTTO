#!/bin/bash
# Does an UNDER-RESOLVED crack favour Green? (question from the ASR / Fourel study, 2026-11-07: there Green won at
# l_c/h < 2.) Same CH two-phase test as run_AT2_cahn_hilliard_kres.sh (k_res = 1e-4, 128^2), but the length scale
# shrunk below the pixel: l = 1/64, 1/128, 1/256, 1/512 -> l/h = 2, 1, 0.5, 0.25 (l/h = 4 = the existing l = 1/32 runs).
# The AT2 strength ~ sqrt(E Gc / l) grows as l shrinks, so LOAD_MAX = 2e-3 sqrt(l0 / l) (l0 = 1/32), 200 steps, the run
# stops once cut through (STOP_FRACTION). Green / Green-Jacobi / dual.
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu GEOMETRY_FILE=geometry/cahn_hilliard_512.npy INCLUSION_STIFFNESS=0.1 \
  GC=1.2857e-8 LAME=0.576923,0.384615 LOAD_DIR=yy LOAD_STEPS=200 RUN_SUFFIX=_ch K_RESIDUAL=1e-4 NPIX=128
job() {  # L P
    lm=$(python3 -c "print(f'{2e-3 * (0.03125 / $1) ** 0.5:.4g}')")
    systemd-run --user --scope -p MemoryMax=4G -q env LENGTH_SCALE=$1 LOAD_MAX=$lm \
        $PY -u example_2D_phase_field_fracture_AT2.py $2 > logs/AT2_N128_$2_incl0.1_l$1_k0.0001_ch.log 2>&1
    echo "$(date +%H:%M:%S) done l=$1 $2"
}
export -f job; export PY
for L in 0.015625 0.0078125 0.00390625 0.001953125; do for P in Green Green_Jacobi dual; do echo "$L $P"; done; done \
    | xargs -P 12 -L 1 bash -c 'job $0 $1'

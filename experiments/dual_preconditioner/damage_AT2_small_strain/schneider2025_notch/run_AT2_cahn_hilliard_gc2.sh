#!/bin/bash
# Cahn-Hilliard two-phase test as in run_AT2_cahn_hilliard_kres.sh (k_res = 1e-4), but phase 2 cracks too, with the
# paper's toughness ratio (Schneider & Kaestner 2025: phase 1 Gc = 2.7 N/mm, phase 2 Gc = 4.7 N/mm -> PHASE2_GC =
# 4.7 / 2.7 = 1.7407) instead of being tough (no crack driving force). Green / Green-Jacobi / dual x 64^2, 128^2 (default),
# CPU; NS="512 256" for the larger grids (overnight 2026-11-03).
# See README.md
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu GEOMETRY_FILE=geometry/cahn_hilliard_512.npy INCLUSION_STIFFNESS=0.1 \
  PHASE2_GC=1.7407 LENGTH_SCALE=0.03125 GC=1.2857e-8 LAME=0.576923,0.384615 LOAD_DIR=yy LOAD_MAX=2e-3 LOAD_STEPS=200 \
  RUN_SUFFIX=_ch K_RESIDUAL=1e-4
job() {  # N P
    systemd-run --user --scope -p MemoryMax=8G -q env NPIX=$1 \
        $PY -u example_2D_phase_field_fracture_AT2.py $2 > logs/AT2_N$1_$2_incl0.1_gc1.7407_l0.03125_k0.0001_ch.log 2>&1
    echo "$(date +%H:%M:%S) done N=$1 $2 (exit $?)"
}
export -f job; export PY
NS=${NS:-128 64}
for N in $NS; do for P in Green Green_Jacobi dual; do echo "$N $P"; done; done | xargs -P 6 -L 1 bash -c 'job $0 $1'

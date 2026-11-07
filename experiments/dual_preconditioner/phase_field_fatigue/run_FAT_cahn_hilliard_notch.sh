#!/bin/bash
# Phase-field fatigue (Carrara et al. 2020, AT2, asymptotic f, alpha_T = Gc / (12 l)) of the notched two-phase
# Cahn-Hilliard cell: R = 0 cyclic uniaxial strain E_yy, amplitude AMP (default 8e-5, ~1/4 of the monotonic peak
# strain 3.4e-4), one load step per cycle at the peak. Fixed l = 1/16 (l/h = 2 / 4 / 8 on 32^2 / 64^2 / 128^2).
# NOTCH_MODE=history: the notch as a phase-field pre-crack (initial history) instead of a soft pixel row.
# Green / Green-Jacobi / dual x NS (default "32 64 128"), CPU, at most 3 runs at a time. See README.md
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu GEOMETRY_FILE=geometry/cahn_hilliard_512.npy \
  INCLUSION_STIFFNESS=0.1 PHASE2_GC=1.7407 NOTCH=0.375,0.625,0.5 LENGTH_SCALE=0.0625 GC=1.2857e-8 \
  LAME=0.576923,0.384615 LOAD_DIR=yy K_RESIDUAL=1e-4 FATIGUE=${FATIGUE:-asymptotic} CYCLES=${CYCLES:-1500} \
  AMPLITUDE=${AMP:-8e-5} MAX_STAGGERED=${MAX_STAGGERED:-500} NOTCH_MODE=${NOTCH_MODE:-stiffness}
job() {  # N P
    systemd-run --user --scope -p MemoryMax=4G -q env NPIX=$1 \
        $PY -u example_2D_phase_field_fatigue_AT2.py $2 > logs/FAT_N$1_$2_amp${AMPLITUDE}_${FATIGUE}$([ "$NOTCH_MODE" = history ] && echo _precrack).log 2>&1
    echo "$(date +%H:%M:%S) done N=$1 $2 (exit $?)"
}
export -f job; export PY
NS=${NS:-32 64 128}
for N in $NS; do for P in ${PRECS:-Green Green_Jacobi dual}; do echo "$N $P"; done; done | xargs -P ${NPROC:-3} -L 1 bash -c 'job $0 $1'

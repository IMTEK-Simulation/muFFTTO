#!/bin/bash
# SUPERSEDED by run_FAT_carrara_beads16.sh: here the cell is 64 l, i.e. l/h = 0.5 / 1 / 2 only (design error).
# Phase-field fatigue with the REAL parameters of Carrara et al. (CMAME 2020, Sec. 4.3, Paris-law set): matrix E = 6 GPa,
# nu = 0.22, Gc = 2.28 N/mm, l = 0.2 mm, alpha_T = Gc / (12 l) = 0.95 N/mm^2, plane stress, AT2, asymptotic f.
# Periodic cell L = 64 l = 12.8 mm (units: length L, stress E_matrix = 6 GPa): l = 1/64 (l/h = 2 / 4 / 8 on 32^2 / 64^2 /
# 128^2), Gc = 2.28 / (6000 * 12.8) = 2.96875e-5, plane-stress Lame lambda* = E nu / (1 - nu^2) = 0.231190, mu = 0.409836.
# Glass beads (E = 70 GPa, nu = 0.22: stiffness 11.6667, tough), 20 % (make_glass_bead_geometry.py); phase-field
# pre-crack x = 0.375 ... 0.625 L (3.2 mm) at y = L/2 (NOTCH_MODE=history). R = 0 cyclic E_yy, amplitude 2.5e-3
# (0.26 x the monotonic peak strain 9.5e-3, logs/mono_beads_N64_GJ.log), one load step per cycle.
# Green / Green-Jacobi / dual x NS (default "32 64 128"), CPU, at most 3 runs at a time. See README.md
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu GEOMETRY_FILE=geometry/glass_beads_512.npy \
  INCLUSION_STIFFNESS=11.6667 NOTCH=0.375,0.625,0.5 NOTCH_MODE=history LENGTH_SCALE=0.015625 GC=2.96875e-5 \
  LAME=0.231190,0.409836 LOAD_DIR=yy K_RESIDUAL=1e-4 FATIGUE=asymptotic CYCLES=1500 AMPLITUDE=${AMP:-2.5e-3} \
  MAX_STAGGERED=500 STOP_FRACTION=0.3 RUN_SUFFIX=_beads
job() {  # N P
    systemd-run --user --scope -p MemoryMax=4G -q env NPIX=$1 \
        $PY -u example_2D_phase_field_fatigue_AT2.py $2 > logs/FAT_beads_N$1_$2_amp${AMPLITUDE}.log 2>&1
    echo "$(date +%H:%M:%S) done N=$1 $2 (exit $?)"
}
export -f job; export PY
NS=${NS:-32 64 128}
for N in $NS; do for P in ${PRECS:-Green Green_Jacobi dual}; do echo "$N $P"; done; done | xargs -P ${NPROC:-3} -L 1 bash -c 'job $0 $1'

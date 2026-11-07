#!/bin/bash
# Separate l/h from the grid size N (all other studies refine N at fixed l, so l/h and N move together):
# CH two-phase AT2 (as run_AT2_cahn_hilliard_small_l.sh, k_res 1e-4) at FIXED l/h = 0.5 and 4 on N = 64 and 256
# (128 = the small-l runs). Green and dual only (no Green-Jacobi on damage). LOAD_MAX = 2e-3 sqrt(l0 / l), l0 = 1/32.
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu GEOMETRY_FILE=geometry/cahn_hilliard_512.npy INCLUSION_STIFFNESS=0.1 \
  GC=1.2857e-8 LAME=0.576923,0.384615 LOAD_DIR=yy LOAD_STEPS=200 RUN_SUFFIX=_ch K_RESIDUAL=1e-4 OUTPUT_FIELDS=damage
job() {  # N L P
    lm=$(python3 -c "print(f'{2e-3 * (0.03125 / $2) ** 0.5:.4g}')")
    systemd-run --user --scope -p MemoryMax=8G -q env NPIX=$1 LENGTH_SCALE=$2 LOAD_MAX=$lm \
        $PY -u example_2D_phase_field_fracture_AT2.py $3 > logs/AT2_N$1_$3_incl0.1_l$2_k0.0001_ch.log 2>&1
    echo "$(date +%H:%M:%S) done N=$1 l=$2 $3"
}
export -f job; export PY
for P in Green dual; do
  echo "64 0.0078125 $P"; echo "64 0.0625 $P"; echo "256 0.001953125 $P"; echo "256 0.015625 $P"
done | xargs -P 8 -L 1 bash -c 'job $0 $1 $2'

#!/bin/bash
# The small grids of the NEMO mesh study (sbatch_fatigue_gpu.sh, case 2D) on the laptop CPU: identical settings,
# 2D 32^2 / 64^2 / 128^2 (l/h = 0.5 / 1 / 2, under-resolved / marginal) x Green / Green-Jacobi / dual, 3 at a time.
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu INCLUSION_STIFFNESS=11.6667 NOTCH=0.375,0.625,0.5 \
  NOTCH_MODE=history LENGTH_SCALE=0.015625 GC=2.96875e-5 LOAD_DIR=yy K_RESIDUAL=1e-4 FATIGUE=asymptotic CYCLES=1500 \
  MAX_STAGGERED=500 STOP_FRACTION=0.3 OUTPUT_FIELDS=damage NC_EVERY=5 DIM=2 GEOMETRY_FILE=geometry/beads2D_2048.npy \
  LAME=0.231190,0.409836 AMPLITUDE=2.0e-3 RUN_SUFFIX=_nemo2D
job() {  # N P
    systemd-run --user --scope -p MemoryMax=4G -q env NPIX=$1 \
        $PY -u example_2D_phase_field_fatigue_AT2.py $2 > logs/FAT_nemo2D_N$1_$2.log 2>&1
    echo "$(date +%H:%M:%S) done N=$1 $2 (exit $?)"
}
export -f job; export PY
NS=${NS:-128 64 32}
for N in $NS; do for P in ${PRECS:-Green Green_Jacobi dual}; do echo "$N $P"; done; done | xargs -P ${NPROC:-3} -L 1 bash -c 'job $0 $1'

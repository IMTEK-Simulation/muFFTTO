#!/bin/bash
# ASR gel expansion, AT2: Green / Green-Jacobi / dual on 32^2, 64^2, 128^2 (fixed l = 1/16, i.e. l/h = 2, 4, 8),
# gel dV/V 0 ... 0.024 in 48 steps, geometry seed 1. Usage: ./run_asr_grid_study.sh "32 64" (grids), at most 3 at once
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export GEOMETRY_NPZ GEL_CELLS RUN_SUFFIX MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu DVV_MAX=${DVV_MAX:-0.024} LOAD_STEPS=${LOAD_STEPS:-48} \
  LENGTH_SCALE=${LENGTH_SCALE:-0.0625}
job() {  # N P
    systemd-run --user --scope -p MemoryMax=4G -q env NPIX=$1 \
        $PY -u asr_AT2_gel_expansion.py $2 > logs/ASR_N$1_$2_l${LENGTH_SCALE}${RUN_SUFFIX}.log 2>&1
    echo "$(date +%H:%M:%S) done N=$1 $2"
}
export -f job; export PY
for N in ${1:-32 64 128}; do for P in Green Green_Jacobi dual; do echo "$N $P"; done; done | xargs -P ${NPAR:-3} -L 1 bash -c 'job $0 $1'

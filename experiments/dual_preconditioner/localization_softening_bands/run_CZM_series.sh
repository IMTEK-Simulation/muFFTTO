#!/bin/bash
# PF-CZM (Wu 2017, Cornelissen) concrete meso-cell 80 mm, b = 5 mm fixed (b/h = 2, 4, 8 at 32^2, 64^2, 128^2):
# 32^2, 64^2, 128^2 x Green / Green_Jacobi / dual, at most 3 processes at once.
# Usage: ./run_CZM_series.sh <LOAD_DIR xx|xy> <K_RESIDUAL> <LOAD_MAX> <LOAD_STEPS>   (extra env passes through)
cd "$(dirname "$0")"
export PY=~/micromamba/envs/muffto-gpu/bin/python
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu LOAD_DIR=${1:-xx} K_RESIDUAL=${2:-1e-3} \
    LOAD_MAX=${3:-3e-3} LOAD_STEPS=${4:-240}
job() {  # N P
    log=logs/CZM_N$1_$2_asr_${LOAD_DIR}_cornelissen_b5mm_k$(python3 -c "print(f'{float(\"$K_RESIDUAL\"):g}')")${RUN_SUFFIX}.log
    systemd-run --user --scope -p MemoryMax=4G -q env NPIX=$1 $PY -u example_2D_pfczm_concrete.py $2 > $log 2>&1
    echo "$(date +%H:%M:%S) done N=$1 $2 $LOAD_DIR k=$K_RESIDUAL"
}
export -f job
for N in 128 64 32; do for P in Green Green_Jacobi dual; do echo "$N $P"; done; done | xargs -P 3 -L 1 bash -c 'job $0 $1'

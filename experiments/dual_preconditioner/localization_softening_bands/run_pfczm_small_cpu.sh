#!/bin/bash
# PF-CZM mesh study, small 2D grids (32-128², tension and shear) on the laptop CPU, 3 processes at once
cd "$(dirname "$0")"
export PY=~/micromamba/envs/muffto-gpu/bin/python MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu K_RESIDUAL=1e-5
job() {  # N P DIR
    lm=3e-3; [ "$3" = xy ] && lm=6e-3
    systemd-run --user --scope -p MemoryMax=4G -q env NPIX=$1 LOAD_DIR=$3 LOAD_MAX=$lm $PY -u pfczm_concrete.py $2 \
        > logs/PFCZM2D_N$1_$2_$3_b1.5mm_k1e-05.log 2>&1
    echo "$(date +%H:%M:%S) done N=$1 $2 $3"
}
export -f job
for D in xx xy; do for N in 128 64 32; do for P in Green Green_Jacobi dual; do echo "$N $P $D"; done; done; done \
    | xargs -P 3 -L 1 bash -c 'job $0 $1 $2'

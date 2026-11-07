#!/bin/bash
# Implicit-gradient damage localisation: 32^2, 64^2, 128^2 x Green / Green_Jacobi / dual, fixed l = 1/16
# (l/h = 2, 4, 8), stiff tough inclusions 10x (polydisperse_pores geometry, 25 %).
# Usage: ./run_GD_series.sh <LOAD_DIR xx|xy> <K_RESIDUAL> [LOAD_MAX] [LOAD_STEPS]   (at most 3 processes at once)
cd "$(dirname "$0")"
export PY=~/micromamba/envs/muffto-gpu/bin/python
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu LENGTH_SCALE=0.0625 INCLUSION_STIFFNESS=10 \
    LOAD_DIR=${1:-xx} K_RESIDUAL=${2:-1e-3} LOAD_MAX=${3:-0.012} LOAD_STEPS=${4:-120}
job() {  # N P
    log=logs/GD_N$1_$2_${LOAD_DIR}_l0.0625_b100_k$(python3 -c "print(f'{float(\"$K_RESIDUAL\"):g}')").log
    systemd-run --user --scope -p MemoryMax=4G -q env NPIX=$1 $PY -u example_2D_gradient_damage_localization.py $2 \
        > $log 2>&1
    echo "$(date +%H:%M:%S) done N=$1 $2 $LOAD_DIR k=$K_RESIDUAL"
}
export -f job
for N in 128 64 32; do for P in Green Green_Jacobi dual; do echo "$N $P"; done; done | xargs -P 3 -L 1 bash -c 'job $0 $1'

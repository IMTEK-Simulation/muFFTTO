#!/bin/bash
# Several fibre realisations at once (the paper: 8 structures at 256^2), Green and dual for every seed, all in
# parallel (CPU, serial, single thread each); settings as run_fibers_small.sh.
# usage:  ./run_fibers_seeds.sh 256 "1 2 3 4 5 6 7"
cd "$(dirname "$0")"
N=${1:-256}
SEEDS=${2:-"1 2 3 4 5 6 7"}
PY=~/micromamba/envs/muffto-gpu/bin/python
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
mkdir -p logs
for SEED in $SEEDS; do
    for P in ${PRECS:-Green dual}; do
        FIBERS=1 FIBER_SEED=$SEED NPIX=$N LENGTH_SCALE=0.01171875 GC=1.953125e-4 K_RESIDUAL=1e-4 \
            LOAD_MAX=0.016 LOAD_STEPS=16 \
            $PY -u example_2D_phase_field_fracture_AT2.py $P > logs/AT2_fibers_s${SEED}_N${N}_${P}.log 2>&1 &
    done
done
wait
grep -H "Elapsed" logs/AT2_fibers_s*_N${N}_*.log

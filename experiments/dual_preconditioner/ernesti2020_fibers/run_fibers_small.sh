#!/bin/bash
# Ernesti, Schneider & Boehlke (2020) Sec. 4.4 reproduction, small grids: random UD fibres (FIBER_SEED), uniaxial strain
# E_xx in 0.1 % steps to 1.6 %, l = 2 eta = 6 um, Gc = 0.1 N/mm, k0 = 1e-4 (units GPa, cell edge 512 um).
# The three elastic preconditioners in parallel (CPU, serial, single thread each).
# usage:  [PRECS="Green dual"] ./run_fibers_small.sh 128 0        (grid, fibre seed)
cd "$(dirname "$0")"
N=${1:-128}
SEED=${2:-0}
PY=~/micromamba/envs/muffto-gpu/bin/python
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
mkdir -p logs
for P in ${PRECS:-Green Green_Jacobi dual}; do
    FIBERS=1 FIBER_SEED=$SEED NPIX=$N LENGTH_SCALE=0.01171875 GC=1.953125e-4 K_RESIDUAL=1e-4 \
        LOAD_MAX=0.016 LOAD_STEPS=16 \
        $PY -u example_2D_phase_field_fracture_AT2.py $P > logs/AT2_fibers_s${SEED}_N${N}_${P}.log 2>&1 &
done
wait
grep -h "Elapsed" logs/AT2_fibers_s${SEED}_N${N}_*.log

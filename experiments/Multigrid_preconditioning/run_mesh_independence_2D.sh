#!/usr/bin/env bash
# Is the objective mesh-size independent? 2D check with stop_criterion_2D.py (energy objective, nu* -0.5, scale 0.3,
# same Cahn-Hilliard start on every grid, 600 iterations):
#   series "2h"     eta = 2 h     (interface shrinks with the grid; what the 3D chain does)   N 32 64 128 256
#   series "eta16"  eta = 1/16    (fixed physical interface width = 2 h of 32^2)              N 32 64 128 256
#   series "eta32"  eta = 1/32    (fixed, = 2 h of 64^2)                                      N 64 128 256
# -> logs/mesh_independence/N<N>_nu-0.5_s0.3_<series>.{npz,txt}; figure: plot_mesh_independence_2D.py
cd "$(dirname "$0")"; source ./fable_env.sh
OUT=logs/mesh_independence; mkdir -p $OUT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OUT_NAME=mesh_independence TARGET_NU=-0.5 TARGET_SCALE=0.3 MAXITER=${MAXITER:-600}
run() { NPIX=$1 ETA=$2 TAG=_$3 $PY -u stop_criterion_2D.py > $OUT/run_N$1_$3.log 2>&1 & }
for N in 32 64 128 256; do run $N $(awk "BEGIN{print 2 / $N}") 2h; done
for N in 64 128 256; do run $N 0.0625 eta16; done
for N in 128 256; do run $N 0.03125 eta32; done
wait

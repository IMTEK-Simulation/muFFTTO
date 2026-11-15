#!/usr/bin/env bash
# The 256^2 runs of run_mesh_independence_2D.sh on the GPU (single-threaded CPU: ~20-30 s per evaluation, GPU ~5 s)
cd "$(dirname "$0")"; source ./fable_env.sh
OUT=logs/mesh_independence; mkdir -p $OUT
export DEVICE=gpu OUT_NAME=mesh_independence TARGET_NU=-0.5 TARGET_SCALE=0.3 MAXITER=${MAXITER:-600}
run() { NPIX=$1 ETA=$2 TAG=_$3 $PY -u stop_criterion_2D.py > $OUT/run_N$1_$3.log 2>&1 & }
N=${NPIX:-256}
run $N $(awk "BEGIN{print 2 / $N}") 2h; run $N 0.0625 eta16; run $N 0.03125 eta32
wait

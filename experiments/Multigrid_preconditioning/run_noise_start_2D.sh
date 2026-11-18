#!/usr/bin/env bash
# 2D noise-start test of the L-BFGS first-step fix: one stream per FIRST_STEP value, grids 256 -> 512 -> 1024
#   FIRST_STEP=<x> [NOISE_N=<M>] ./run_noise_start_2D.sh [grids...]   -> logs/noise_start_2D/run_N<N>_noise<M>_first<x>.log
cd "$(dirname "$0")"; source ./fable_env.sh
export DEVICE=${DEVICE:-gpu} MAXITER=${MAXITER:-300} FIRST_STEP=${FIRST_STEP:-0} SEED=${SEED:-0}
mkdir -p logs/noise_start_2D
for N in ${@:-256 512 1024}; do
  NPIX=$N $PY -u noise_start_2D.py > logs/noise_start_2D/run_N${N}_noise${NOISE_N:-$N}_first${FIRST_STEP}.log 2>&1
done

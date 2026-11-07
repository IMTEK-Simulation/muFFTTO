#!/bin/bash
# Run fourel_nonlocal_damage.py on the laptop CPU (single-threaded BLAS, JAX on CPU), memory capped.
#   ./run_fourel.sh <solver> <N> [ENV=VALUE ...]      e.g. ./run_fourel.sh MS 128 TOL_MS=1e-6
# log: logs/FOUREL_N<N>_<solver>_<env tags>.log
cd "$(dirname "$0")"
solver=$1; N=$2; shift 2
tag=$(echo "$@" | tr ' =' '_-')
log=logs/FOUREL_N${N}_${solver}${tag:+_$tag}.log
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
    NPIX=$N "$@" systemd-run --user --scope -q -p MemoryMax=12G \
    ~/micromamba/envs/muffto-gpu/bin/python fourel_nonlocal_damage.py "$solver" > "$log" 2>&1

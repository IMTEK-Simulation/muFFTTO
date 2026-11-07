#!/bin/bash
# usage: ./run_pf_fourel.sh <VARIANT> <NPIX> <Green|Green_Jacobi|dual> [extra env assignments...]
# runs asr_pf_fourel.py in a memory-limited systemd scope, log to logs/pf/<run>.log
cd "$(dirname "$0")"
V=$1; N=$2; P=$3; shift 3
mkdir -p logs/pf
env JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VARIANT=$V NPIX=$N "$@" \
  systemd-run --user --scope -q -p MemoryMax=12G \
  ~/micromamba/envs/muffto-gpu/bin/python asr_pf_fourel.py $P \
  > logs/pf/ASRpf_${V}_N${N}_${P}${RUN_SUFFIX}$(printf '%s' "$*" | grep -o 'RUN_SUFFIX=[^ ]*' | cut -d= -f2).log 2>&1

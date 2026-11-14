#!/bin/bash
# Sintering mesh study: fixed physical surface width W = 4 nm, grids 32/64/128/256 (W/h = 1/2/4/8),
# Green / Green_Jacobi / dual, same physics (dt = 1e-7 s, 5000 steps = 500 us, elastic solves every 25 steps).
# usage: run_sintering_study.sh <NPIX> <cpu|gpu> [preconditioners...]
# Each run in its own memory-capped systemd scope, background; logs/ and exp_data/.
cd "$(dirname "$0")"
N=$1; DEV=$2; shift 2
PRECS=${@:-Green Green_Jacobi dual}
PY=~/micromamba/envs/muffto-gpu/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export CUDA_PATH=~/micromamba/envs/muffto-gpu/targets/x86_64-linux
[ "$DEV" = cpu ] && export JAX_PLATFORMS=cpu
for P in $PRECS; do
  NPIX=$N DEVICE=$DEV DT=${DT:-1e-7} STEPS=${STEPS:-5000} MECH_EVERY=${MECH_EVERY:-25} SNAP=${SNAP:-50} \
    nohup systemd-run --user --scope -p MemoryMax=${MEMMAX:-6G} $PY sintering_phase_field.py $P \
    > logs/SINT_N${N}_${P}_k${K_RES:-1e-4}_P${PRESSURE:-50e6}${RUN_SUFFIX}.log 2>&1 &
done

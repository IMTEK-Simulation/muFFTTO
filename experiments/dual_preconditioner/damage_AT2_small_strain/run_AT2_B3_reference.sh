#!/bin/bash
# benchmark B3 reference (2026-10-31): stiff tough inclusions 10x, fixed l = 1/32, dual preconditioner;
# 256^2 on the CPU (4 MPI ranks) and 512^2 on the GPU in parallel; sleep blocked (a suspend hangs CUDA jobs)
cd "$(dirname "$0")"
export MPLBACKEND=Agg OMP_NUM_THREADS=1
PY=~/micromamba/envs/muffto-gpu/bin/python
systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block sleep infinity &
INHIBIT=$!
COMMON="MICROSTRUCTURE=polydisperse_pores INCLUSION_STIFFNESS=10 LENGTH_SCALE=0.03125"
env $COMMON JAX_PLATFORMS=cpu NPIX=256 ~/micromamba/envs/muffto-gpu/bin/mpirun -n 4 --bind-to none \
    $PY -u example_2D_phase_field_fracture_AT2.py dual > logs/AT2_N256_dual_polydisperse_pores_incl10_l0.03125.log 2>&1 &
env $COMMON JAX_PLATFORMS=cpu CUDA_PATH=~/micromamba/envs/muffto-gpu/targets/x86_64-linux \
    XLA_PYTHON_CLIENT_PREALLOCATE=false DEVICE=gpu NPIX=512 \
    $PY -u example_2D_phase_field_fracture_AT2.py dual > logs/AT2_N512_dual_polydisperse_pores_incl10_l0.03125.log 2>&1 &
wait %2 %3
kill $INHIBIT

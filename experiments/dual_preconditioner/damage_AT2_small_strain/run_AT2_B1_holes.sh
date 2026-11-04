#!/bin/bash
# benchmark B1 hole variant (2026-10-31): one centred circular hole (zero stiffness, no damage field), fixed l = 1/32,
# Green-Jacobi; 32^2, 128^2 serial and 256^2 (4 MPI ranks) on the CPU, 512^2 on the GPU, all in parallel
# (64^2 was the test run); sleep blocked for the GPU run
cd "$(dirname "$0")"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 HOLES=1 LENGTH_SCALE=0.03125
PY=~/micromamba/envs/muffto-gpu/bin/python
systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block sleep infinity &
INHIBIT=$!
for n in 32 128; do
    JAX_PLATFORMS=cpu NPIX=$n $PY -u example_2D_phase_field_fracture_AT2.py Green_Jacobi \
        > logs/AT2_N${n}_Green_Jacobi_holes_l0.03125.log 2>&1 &
done
JAX_PLATFORMS=cpu NPIX=256 ~/micromamba/envs/muffto-gpu/bin/mpirun -n 4 --bind-to none \
    $PY -u example_2D_phase_field_fracture_AT2.py Green_Jacobi > logs/AT2_N256_Green_Jacobi_holes_l0.03125.log 2>&1 &
JAX_PLATFORMS=cpu CUDA_PATH=~/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false \
    DEVICE=gpu NPIX=512 $PY -u example_2D_phase_field_fracture_AT2.py Green_Jacobi \
    > logs/AT2_N512_Green_Jacobi_holes_l0.03125.log 2>&1 &
wait %2 %3 %4 %5
kill $INHIBIT

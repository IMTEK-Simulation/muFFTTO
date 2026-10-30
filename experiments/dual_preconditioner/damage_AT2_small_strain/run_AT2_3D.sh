#!/bin/bash
# 3D AT2 (polydisperse ball pores, hybrid g(d) C_0, trilinear hexahedra): Green / Green-Jacobi / dual elastic CG
# 16^3 on the CPU (the three in parallel; JAX_PLATFORMS=cpu: else JAX takes the GPU), 32^3 on the GPU (one after the other); dual with DUAL_INVERSE=fly
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export MICROSTRUCTURE=polydisperse_pores DIM=3 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
for P in Green Green_Jacobi dual; do
  systemd-run --user --scope -p MemoryMax=4G -q env NPIX=16 JAX_PLATFORMS=cpu $PY example_2D_phase_field_fracture_AT2.py $P \
    > logs/AT2_3D_N16_$P.log 2>&1 &
done
systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block bash -c "
for P in dual Green Green_Jacobi; do
  systemd-run --user --scope -p MemoryMax=10G -q env NPIX=32 DEVICE=gpu \
    CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false \
    $PY example_2D_phase_field_fracture_AT2.py \$P > logs/AT2_3D_N32_gpu_\$P.log 2>&1
done"
wait

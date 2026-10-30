#!/bin/bash
# 3D AT2 (polydisperse ball pores, hybrid g(d) C_0, trilinear hexahedra) at FIXED length scale l = 0.0625
# (2 voxels at 32^3, 3 at 48^3, 4 at 64^3): Green / Green-Jacobi / dual elastic CG on the GPU, one run after the other.
# dual with DUAL_INVERSE=fly; MPLBACKEND=Agg: plt.show() does not block the batch
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export MICROSTRUCTURE=polydisperse_pores DIM=3 LENGTH_SCALE=0.0625 DEVICE=gpu MPLBACKEND=Agg OMP_NUM_THREADS=1 \
  CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false
for N in 32 48 64; do
  for P in dual Green_Jacobi Green; do
    echo "$(date +%H:%M:%S) start N=$N $P"
    systemd-run --user --scope -p MemoryMax=20G -q env NPIX=$N $PY example_2D_phase_field_fracture_AT2.py $P \
      > logs/AT2_3D_N${N}_gpu_l0.0625_$P.log 2>&1
    echo "$(date +%H:%M:%S) done  N=$N $P (exit $?)"
  done
done

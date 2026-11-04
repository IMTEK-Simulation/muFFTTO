#!/bin/bash
# overnight 2026-10-31, after run_overnight_3D_gpu.sh: 3D stiff tough inclusions (10x), 96^3 at l/h = 2 (l = 1/48), dual,
# GPU (~5.7 GB of 8; ~6.2 KB per voxel measured)
cd "$(dirname "$0")"
while pgrep -f run_overnight_3D_gpu.sh >/dev/null; do sleep 60; done
PY=~/micromamba/envs/muffto-gpu/bin/python
export MICROSTRUCTURE=polydisperse_pores DIM=3 DEVICE=gpu MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false
L=$($PY -c 'print(1/48)')
echo "$(date +%H:%M:%S) start N=96 l=$L incl=10 dual"
systemd-run --user --scope -p MemoryMax=24G -q env NPIX=96 LENGTH_SCALE=$L INCLUSION_STIFFNESS=10 \
    $PY -u example_2D_phase_field_fracture_AT2.py dual > logs/AT2_N96_dual_polydisperse_pores_incl10_l0.0208333_3D.log 2>&1
echo "$(date +%H:%M:%S) done  N=96 dual (exit $?)"

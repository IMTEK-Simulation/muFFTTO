#!/bin/bash
# Cahn-Hilliard two-phase problem of run_AT2_cahn_hilliard_kres.sh (same parameters, fixed l = 1/32: l/h = 8 / 16) at
# k_res = 1e-4 on larger grids, GPU, one run after another: 256^2 dual, Green-Jacobi, Green, then 512^2 dual,
# Green-Jacobi. Waits for the hybrid green_dual runs to finish first. See README.md
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
while pgrep -f "phase_field_fracture_AT2[.]py green_dual" > /dev/null; do sleep 30; done
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu GEOMETRY_FILE=geometry/cahn_hilliard_512.npy INCLUSION_STIFFNESS=0.1 \
  LENGTH_SCALE=0.03125 GC=1.2857e-8 LAME=0.576923,0.384615 LOAD_DIR=yy LOAD_MAX=2e-3 LOAD_STEPS=200 RUN_SUFFIX=_ch \
  K_RESIDUAL=1e-4 DEVICE=gpu CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false
for NP in "256 dual" "256 Green_Jacobi" "256 Green" "512 dual" "512 Green_Jacobi"; do
    set -- $NP
    echo "$(date +%H:%M:%S) start N=$1 $2"
    systemd-run --user --scope -p MemoryMax=16G -q env NPIX=$1 \
        $PY -u example_2D_phase_field_fracture_AT2.py $2 > logs/AT2_N$1_$2_incl0.1_l0.03125_k0.0001_ch.log 2>&1
    echo "$(date +%H:%M:%S) done  N=$1 $2 (exit $?)"
done

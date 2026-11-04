#!/bin/bash
# Hybrid preconditioner 'green_dual' (Green until max d >= 0.95, then the dual) on the Cahn-Hilliard two-phase problem
# of run_AT2_cahn_hilliard_kres.sh (same parameters), GPU, one run after another: k_res = 1e-4, 1e-5, 1e-6 x 128^2, 64^2.
# See README.md
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu GEOMETRY_FILE=geometry/cahn_hilliard_512.npy INCLUSION_STIFFNESS=0.1 \
  LENGTH_SCALE=0.03125 GC=1.2857e-8 LAME=0.576923,0.384615 LOAD_DIR=yy LOAD_MAX=2e-3 LOAD_STEPS=200 RUN_SUFFIX=_ch \
  DEVICE=gpu CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false
for N in 128 64; do for K in 1e-4 1e-5 1e-6; do
    k=$(python3 -c "print(f'{float(\"$K\"):g}')")
    echo "$(date +%H:%M:%S) start N=$N k=$K"
    systemd-run --user --scope -p MemoryMax=16G -q env NPIX=$N K_RESIDUAL=$K \
        $PY -u example_2D_phase_field_fracture_AT2.py green_dual > logs/AT2_N${N}_green_dual_incl0.1_l0.03125_k${k}_ch.log 2>&1
    echo "$(date +%H:%M:%S) done  N=$N k=$K (exit $?)"
done; done

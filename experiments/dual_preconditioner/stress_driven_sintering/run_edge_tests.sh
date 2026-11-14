#!/bin/bash
# 50 MPa GB-profile runs: third interface width (W = 2 nm, N = 192, same W/h) and 4x surface diffusion (W = 4, N = 96).
# usage: run_edge_tests.sh
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export OMP_NUM_THREADS=2 JAX_PLATFORMS=cpu
common="DEVICE=cpu DT=1e-8 STEPS=40000 SNAP=10000 MECH_EVERY=2500 CHECKPOINT_SECONDS=100000 PRESSURE=50e6 CHAIN=1 RB_TAU=1e-7 GEOMETRY=geometry/two_chain.npz"
env $common NPIX=192 WIDTH=2 RUN_SUFFIX=_W2profile systemd-run --user --scope -q -p MemoryMax=8G $PY sintering_phase_field.py Green > logs/chain_W2_profile.log 2>&1 &
env $common NPIX=96 SURF_FACTOR=4 RUN_SUFFIX=_surf4profile systemd-run --user --scope -q -p MemoryMax=4G $PY sintering_phase_field.py Green > logs/chain_surf4_profile.log 2>&1 &
wait

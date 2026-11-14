#!/bin/bash
# Stage 2 convergence at 50 MPa, dt = 1e-8 s: t_r limit (N = 96) and mesh check (N = 128), laptop CPU.
# usage: run_chain_convergence.sh
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export OMP_NUM_THREADS=2 JAX_PLATFORMS=cpu
run() {  # N tau
  NPIX=$1 DEVICE=cpu DT=1e-8 STEPS=200000 SNAP=5000 MECH_EVERY=2500 CHECKPOINT_SECONDS=100000 PRESSURE=50e6 CHAIN=1 \
    RB_TAU=$2 GEOMETRY=geometry/two_chain.npz RUN_SUFFIX=_chain_P50_tau$2_dt1e-8 \
    systemd-run --user --scope -q -p MemoryMax=5G $PY sintering_phase_field.py Green \
    > logs/chain_N$1_P50_tau$2_dt1e-8.log 2>&1 &
}
run 96 1e-6
run 96 5e-7
run 96 1e-7
run 128 2e-7
wait

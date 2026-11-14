#!/bin/bash
# Stage 2: 2-grain periodic chain in a contracting cell, GB load = resolved elastic stress (laptop CPU).
# usage: run_chain_pressure.sh
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export OMP_NUM_THREADS=2 JAX_PLATFORMS=cpu
run() {  # P[MPa] tau[s] tag extra-env
  env NPIX=96 DEVICE=cpu DT=1e-7 STEPS=20000 SNAP=500 MECH_EVERY=250 PRESSURE=${1}e6 CHAIN=1 RB_TAU=$2 \
    GEOMETRY=geometry/two_chain.npz RUN_SUFFIX=_chain_$3 $4 \
    systemd-run --user --scope -q -p MemoryMax=4G $PY sintering_phase_field.py Green > logs/chain_$3.log 2>&1 &
}
run 0 0 noRB SKIP_MECH=1
run 0 2e-6 P0 SKIP_MECH=1
run 25 2e-6 P25
run 50 2e-6 P50
run 100 2e-6 P100
wait

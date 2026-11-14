#!/bin/bash
# GB mu profile runs (mu saved in snapshots) at several pressures, same settings as the 50 MPa profile run (laptop CPU).
# usage: run_profile_pressure.sh
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export OMP_NUM_THREADS=2 JAX_PLATFORMS=cpu
run() {  # P[MPa] extra-env
  env NPIX=96 DEVICE=cpu DT=1e-8 STEPS=40000 SNAP=10000 MECH_EVERY=2500 CHECKPOINT_SECONDS=100000 PRESSURE=${1}e6 CHAIN=1 \
    RB_TAU=1e-7 GEOMETRY=geometry/two_chain.npz RUN_SUFFIX=_profile $2 \
    systemd-run --user --scope -q -p MemoryMax=4G $PY sintering_phase_field.py Green > logs/chain_profile_P$1.log 2>&1 &
}
run 0
run 25
run 100
wait

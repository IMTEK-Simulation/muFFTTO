#!/bin/bash
# Stage 1: two particles, Seiz-type rigid-body approach, uniform pressure bias, no elastic solve (laptop CPU).
# usage: run_rb_pressure.sh
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export OMP_NUM_THREADS=2 JAX_PLATFORMS=cpu
run() {  # P[MPa] tau[s]
  NPIX=128 DEVICE=cpu DT=1e-7 STEPS=20000 SNAP=500 MECH_EVERY=500 PRESSURE=0 SKIP_MECH=1 RB_TAU=$2 RB_PRESSURE=${1}e6 \
    GEOMETRY=geometry/two_particles.npz RUN_SUFFIX=_rb_P$1_tau$2 \
    systemd-run --user --scope -q -p MemoryMax=4G $PY sintering_phase_field.py Green > logs/rb_P$1_tau$2.log 2>&1 &
}
for P in 0 50 100 200; do run $P 2e-6; done
run 100 1e-6
run 100 2e-5
wait

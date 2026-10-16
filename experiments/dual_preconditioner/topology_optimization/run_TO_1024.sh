#!/bin/bash
# 1024^2 topology optimization (dual, soft 1e-3, energy CG stop, filter 1/64, start = the 256^2 serial start), 16 MPI
# processes, once run_TO_queue.py has finished; every 10th iterate goes to the history file (for a video).
cd "$(dirname "$0")"
LOGS="$PWD/logs"
until tail -1 "$LOGS"/queue.log | grep -q "queue finished"; do sleep 60; done
echo "$(date +%H:%M:%S) start N1024 soft1e-3 dual (mpiexec -n 16)" >> "$LOGS"/queue.log
NPIX=1024 SOFT_EXP=3 ETA=0.015625 CG_MODE=energy SHOW_PLOTS=0 GTOL=0 HISTORY_EVERY=10 INIT_NOISE_GRID=256 MPLBACKEND=Agg \
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 ~/micromamba/envs/muffto-gpu/bin/mpiexec -n 16 \
    ~/micromamba/envs/muffto-gpu/bin/python -u example_2D_elasticity_TO.py dual > "$LOGS"/TO_N1024_soft1e-3_dual.log 2>&1
echo "$(date +%H:%M:%S) done  N1024 soft1e-3 dual rc=$?" >> "$LOGS"/queue.log

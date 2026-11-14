#!/bin/bash
# 512^2 and 1024^2 sintering runs on the laptop GPU, W = 10 nm and W = 4 nm, Green / Green_Jacobi / dual.
# 512^2 first (6 jobs together), then 1024^2 (6 jobs). Same settings as run_sintering_study.sh.
cd "$(dirname "$0")"
for N in 512 1024; do
  WIDTH=10 RUN_SUFFIX=_W10 ./run_sintering_study.sh $N gpu
  WIDTH=4 RUN_SUFFIX= ./run_sintering_study.sh $N gpu
  sleep 60
  while [ "$(ps -C python -o args= | grep -c "[s]intering_phase_field")" -gt 0 ]; do sleep 120; done
  echo "$(date +%H:%M) done N=$N"
done

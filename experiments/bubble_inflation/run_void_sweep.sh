#!/bin/bash
# void-thickness sweep, wall-FFT corrected Green, clamped pressurised-hole plate (plate 0.875 fixed, cell grows)
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
for n in 32 64 128; do
  for t in 0.03125 0.0625 0.125 0.25; do
    [ "$n" = 32 ] && [ "$t" = 0.0625 ] && continue          # done (hole_G_wall_t0.0625_n32.log)
    JAX_PLATFORMS=cpu NNN=$n VOID_T=$t PRECOND=G_wall $PY pressurized_hole.py > hole_G_wall_t${t}_n${n}.log 2>&1 &
  done
  wait
done

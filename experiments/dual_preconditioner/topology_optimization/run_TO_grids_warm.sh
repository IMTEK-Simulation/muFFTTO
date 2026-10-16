#!/bin/bash
# Green-Jacobi and dual, soft 1e-3 and 1e-5, eta = 1/32, warm-start energy stop: the grids given as arguments one
# after the other (all four runs of a grid in parallel, the next grid when they are done).
#   nohup ./run_TO_grids_warm.sh 512 1024 &      -> logs/TO_N<N>_soft1e-<e>_eta1-32_warm_<prec>.log
cd "$(dirname "$0")"
for n in "$@"; do
  # wait for any running run of this script's settings (e.g. a grid started by hand)
  while ps -eo args | grep -q "[e]xample_2D_elasticity_TO.py"; do sleep 60; done
  echo "$(date +%H:%M:%S) start N$n" >> logs/grids_warm.log
  for e in 3 5; do
    for p in Green_Jacobi dual; do
      NPIX=$n SOFT_EXP=$e ETA=0.03125 CG_MODE=energy GTOL=0 SHOW_PLOTS=0 MPLBACKEND=Agg JAX_PLATFORMS=cpu \
      OMP_NUM_THREADS=3 ~/micromamba/envs/muffto-gpu/bin/python -u example_2D_elasticity_TO.py $p \
          > logs/TO_N${n}_soft1e-${e}_eta1-32_warm_$p.log 2>&1 &
    done
  done
  wait
  echo "$(date +%H:%M:%S) done  N$n" >> logs/grids_warm.log
done

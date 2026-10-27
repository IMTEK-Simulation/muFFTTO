#!/bin/bash
# TO grid series with the sharper interface eta = 0.014 (double well D = 1, stress weight 5), on the GPU:
# 128^2, 256^2, 512^2 one after the other; per grid Green-Jacobi and dual, soft 1e-3 and 1e-5 (4 runs at once).
#   nohup ./run_TO_eta0014_gpu.sh &      -> logs/TO_N<N>_soft1e-<e>_eta0.014_warm_<prec>.log
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu CUDA_PATH=~/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false
for n in 128 256 512; do
  echo "$(date +%H:%M:%S) start N$n eta 0.014 (GPU)" >> logs/grids_eta0014.log
  for e in 3 5; do
    for p in Green_Jacobi dual; do
      DEVICE=gpu NPIX=$n SOFT_EXP=$e ETA=0.014 CG_MODE=energy GTOL=0 SHOW_PLOTS=0 MPLBACKEND=Agg OMP_NUM_THREADS=2 \
        ~/micromamba/envs/muffto-gpu/bin/python -u example_2D_elasticity_TO.py $p > logs/TO_N${n}_soft1e-${e}_eta0.014_warm_$p.log 2>&1 &
    done
  done
  wait
  echo "$(date +%H:%M:%S) done  N$n eta 0.014 (GPU)" >> logs/grids_eta0014.log
done

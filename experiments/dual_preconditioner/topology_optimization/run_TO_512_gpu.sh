#!/bin/bash
# TO 512^2 on the GPU: Green-Jacobi and dual, soft 1e-3 and 1e-5, eta = 1/32, warm-start energy stop, all four at
# once on the one GPU; waits until the AT2 GPU runs (../damage_AT2_small_strain/run_AT2_512_gpu.sh) are finished
#   nohup ./run_TO_512_gpu.sh &      -> logs/TO_N512_soft1e-<e>_eta1-32_warm_<prec>.log
cd "$(dirname "$0")"
while ps -eo args | grep -q "[r]un_AT2_512_gpu.sh"; do sleep 60; done
echo "$(date +%H:%M:%S) start N512 (GPU)" >> logs/grids_warm.log
export JAX_PLATFORMS=cpu CUDA_PATH=~/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false
for e in 3 5; do
  for p in Green_Jacobi dual; do
    DEVICE=gpu NPIX=512 SOFT_EXP=$e ETA=0.03125 CG_MODE=energy GTOL=0 SHOW_PLOTS=0 MPLBACKEND=Agg OMP_NUM_THREADS=2 \
      ~/micromamba/envs/muffto-gpu/bin/python -u example_2D_elasticity_TO.py $p > logs/TO_N512_soft1e-${e}_eta1-32_warm_$p.log 2>&1 &
  done
done
wait
echo "$(date +%H:%M:%S) done  N512 (GPU)" >> logs/grids_warm.log

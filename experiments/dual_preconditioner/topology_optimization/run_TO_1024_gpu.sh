#!/bin/bash
# TO 1024^2 on the GPU, soft 1e-3 only: dual and Green-Jacobi, eta = 1/32, warm-start energy stop.
# A 1024^2 run needs about 4 GB on the 8 GB card: dual starts when the soft 1e-3 512^2 runs are done, Green-Jacobi as
# soon as the GPU has 4.5 GB free (e.g. when the soft 1e-5 512^2 runs are done).
#   nohup ./run_TO_1024_gpu.sh &      -> logs/TO_N1024_soft1e-3_eta1-32_warm_<prec>.log
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu CUDA_PATH=~/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false
start() {
  echo "$(date +%H:%M:%S) start N1024 soft1e-3 $1 (GPU)" >> logs/grids_warm.log
  DEVICE=gpu NPIX=1024 SOFT_EXP=3 ETA=0.03125 CG_MODE=energy GTOL=0 SHOW_PLOTS=0 MPLBACKEND=Agg OMP_NUM_THREADS=2 \
    ~/micromamba/envs/muffto-gpu/bin/python -u example_2D_elasticity_TO.py $1 > logs/TO_N1024_soft1e-3_eta1-32_warm_$1.log 2>&1 &
}
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
# soft 1e-3 512^2 runs still going?
while ps -eo args,env 2>/dev/null | grep -q "[e]xample_2D_elasticity_TO" && \
      grep -L "Data saved" logs/TO_N512_soft1e-3_eta1-32_warm_{dual,Green_Jacobi}.log | grep -q .; do sleep 60; done
start dual
sleep 600                      # let dual allocate its memory first
until [ "$(free_mb)" -gt 4500 ]; do sleep 60; done
start Green_Jacobi
wait
echo "$(date +%H:%M:%S) done  N1024 soft1e-3 (GPU)" >> logs/grids_warm.log

#!/bin/bash
# simplified material at NX = 128 (where the paper's material matches the paper to ~15 %): inclusion contrast 10 with
# Lambda/mu 100 and 10; true holes, 1x2 cells, SRI Q1, TR Newton + Green, GPU, X compression to eps 0.05
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
COMMON="-n 128 -inc 10 --eps_max 0.05 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --precond green --device gpu --frames"
for L in 100 10; do
  systemd-run --user --scope -p MemoryMax=10G -q env CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux \
    XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg OMP_NUM_THREADS=2 \
    $PY stefanus_2D_finite_strain_NeoHookean_TR.py $COMMON --incl_contrast 10 --lam_over_mu $L --tag _c10_L$L \
    > exp_data/log_simplify128_c10_L$L.txt 2>&1 &
done
wait

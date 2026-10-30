#!/bin/bash
# simplified material (inclusion contrast 10, Lambda/mu 100) at NX = 256, true holes, 1x2 cells, SRI Q1, TR Newton +
# Green, GPU, X compression to eps 0.2 (the end of paper Fig. 4), frames for the video
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
systemd-run --user --scope -p MemoryMax=12G -q env CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux \
  XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg OMP_NUM_THREADS=2 \
  $PY stefanus_2D_finite_strain_NeoHookean_TR.py -n 256 -inc 40 --eps_max 0.2 --direction X --cells 1x2 \
  --imperfection 1e-3 --element bilinear_rectangle --sri --precond green --device gpu --frames \
  --incl_contrast 10 --lam_over_mu 100 --tag _c10_L100 > exp_data/log_simplify256_c10_L100.txt 2>&1

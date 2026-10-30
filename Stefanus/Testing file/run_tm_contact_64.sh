#!/bin/bash
# third-medium contact test at 64 px (CPU): simplified material (inclusion contrast 10, Lambda/mu 100), voids filled
# with a third medium k = 1e-4 / 1e-3 (nu = 0.3, HuHu regularization alpha 1e-6 in the voids only), and true holes as
# reference; 1x2 cells, SRI Q1, TR Newton + Green, X compression to eps 0.2 in 40 increments, frames for the video
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
COMMON="-n 64 -inc 40 --eps_max 0.2 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --precond green --incl_contrast 10 --lam_over_mu 100 --frames --verbose"
run() {  # tag, extra args
  systemd-run --user --scope -p MemoryMax=8G -q env OMP_NUM_THREADS=4 MPLBACKEND=Agg JAX_PLATFORMS=cpu \
    $PY stefanus_2D_finite_strain_NeoHookean_TR.py $COMMON $2 --tag _c10_L100$1 > exp_data/log_contact64$1.txt 2>&1
}
run _TM1e-4 "--third_medium 1e-4" &
run _TM1e-3 "--third_medium 1e-3" &
run _holes "" &
wait

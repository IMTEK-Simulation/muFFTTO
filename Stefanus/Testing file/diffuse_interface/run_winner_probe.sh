#!/bin/bash
# probes of the winning set-up (dual ref lambda/mu 10 on the graded third medium), linear test at F = I, GPU:
#   grid 64 / 128 / 256 (CAP 1e-3, w 2); w 1 at 128; softer medium k 1e-5 graded to 1e-3 (w 2) vs sharp, dual and Green
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
ENV="CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg OMP_NUM_THREADS=2"
MAT="--direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --incl_contrast 10 --lam_over_mu 100 --device gpu --interp log -inc 1 --eps_max 0.0005 --linear_test"
W="--precond dual --ref_lam_over_mu 10"
{
for N in 64 128 256; do echo "-n $N --third_medium 1e-4 --diffuse 2 --void_grade 1e-3 $W"; done
echo "-n 256 --third_medium 1e-4 --diffuse 0 $W"
echo "-n 128 --third_medium 1e-4 --diffuse 1 --void_grade 1e-3 $W"
echo "-n 128 --third_medium 1e-5 --diffuse 2 --void_grade 1e-3 $W"
echo "-n 128 --third_medium 1e-5 --diffuse 0 $W"
echo "-n 128 --third_medium 1e-5 --diffuse 0 --precond green"
} | xargs -P 3 -I{} sh -c "env $ENV timeout 2h $PY stefanus_diffuse.py $MAT {} 2>&1 | grep 'CG iterations\|Error' \
    | sed 's/^/{}  /'" > exp_data/log_winner_probe.txt

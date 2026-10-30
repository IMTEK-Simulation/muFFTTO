#!/bin/bash
# combine: dual reference lambda/mu 10 + stress-matched ramp (w 2, c 1.2); all with the fast Hessian (2x2 tangent)
#   (A) GPU, 128 px, linear test: dual ref 10 on sharp and on w 2 c 1.2
#   (B) CPU, 64 px, sharp, 2 increments: Green vs dual (ref 10, Green-norm trust region) -> wall time
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
ENV="CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg OMP_NUM_THREADS=2"
MAT="--direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --incl_contrast 10 --lam_over_mu 100 --third_medium 1e-4 --interp log"
{
echo "-n 128 --device gpu --diffuse 0 --precond dual --ref_lam_over_mu 10 -inc 1 --eps_max 0.0005 --linear_test"
echo "-n 128 --device gpu --diffuse 2 --ramp_shift 1.2 --precond dual --ref_lam_over_mu 10 -inc 1 --eps_max 0.0005 --linear_test"
} | xargs -P 2 -I{} sh -c "env $ENV timeout 2h $PY stefanus_diffuse.py $MAT {} 2>&1 | grep 'CG iterations\|Error' \
    | sed 's/^/{}  /'" > exp_data/log_combo_linear.txt &
run() { OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu MPLBACKEND=Agg $PY stefanus_diffuse.py -n 64 -inc 2 --eps_max 0.01 $MAT --diffuse 0 \
          --verbose $2 --tag _fasthess_$1 > exp_data/log_fasthess_nonlinear64_$1.txt 2>&1; }
run green "--precond green" &
run dual_ref10 "--precond dual --ref_lam_over_mu 10 --tr_metric green" &
wait

#!/bin/bash
# calibrated ramp shifts (stress after one increment = sharp): w 1 c 1.1, w 2 c 1.2; 128 px, GPU, third medium 1e-4.
#   (A) linear test: dual and Green
#   (B) 6 increments to eps 0.03 (through buckling), dual + Green-norm trust region: sharp, w 2 c 1.2
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
ENV="CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg OMP_NUM_THREADS=2"
MAT="-n 128 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --incl_contrast 10 --lam_over_mu 100 --third_medium 1e-4 --device gpu --interp log"
{
for wc in 1,1.1 2,1.2; do W=${wc%,*}; C=${wc#*,}
  for P in dual green; do echo "--diffuse $W --ramp_shift $C --precond $P -inc 1 --eps_max 0.0005 --linear_test"; done
done
echo "--diffuse 0 --precond dual --tr_metric green -inc 6 --eps_max 0.03 --frames --tag _path_sharp"
echo "--diffuse 2 --ramp_shift 1.2 --precond dual --tr_metric green -inc 6 --eps_max 0.03 --frames --tag _path_w2_c1.2"
} | xargs -P 6 -I{} sh -c "env $ENV timeout 4h $PY stefanus_diffuse.py $MAT {} 2>&1 | grep '^inc\|CG iterations\|TR its, \|Error\|failed' \
    | sed 's/^/{}  /'" >> exp_data/log_ramp_calibrated.txt

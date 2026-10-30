#!/bin/bash
# calibrate the ramp position: log profile shifted into the circles by c w h (c = 0 centred: too soft, c = 3 inside:
# too stiff). 128 px, GPU, simplified material, third medium k = 1e-4. For each (w, c): the stress after one increment
# (dual + Green-norm trust region; sharp reference 0.00166 MPa) and the linear-test count of the dual
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
ENV="CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg OMP_NUM_THREADS=2"
MAT="-n 128 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --incl_contrast 10 --lam_over_mu 100 --third_medium 1e-4 --device gpu --precond dual --interp log"
CASES="${CASES:-1,0.75 1,1.5 2,0.75 2,1.5}"
{
for wc in $CASES; do W=${wc%,*}; C=${wc#*,}
  echo "--diffuse $W --ramp_shift $C -inc 1 --eps_max 0.005 --tr_metric green --tag _shift_w${W}_c$C"
  echo "--diffuse $W --ramp_shift $C -inc 1 --eps_max 0.0005 --linear_test"
done
} | xargs -P 4 -I{} sh -c "env $ENV timeout 2h $PY stefanus_diffuse.py $MAT {} 2>&1 | grep '^inc\|CG iterations\|Error' \
    | sed 's/^/{}  /'" >> exp_data/log_ramp_shift.txt

#!/bin/bash
# trust-region radius (Green norm) for the winner: 64 px, CPU, 3 increments through buckling (eps 0.015);
# (delta0, delta_max) = (0.05, 0.5) default, (0.2, 2), (0.5, 5)
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
C="-n 64 -inc 3 --eps_max 0.015 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --incl_contrast 10 --lam_over_mu 100 --third_medium 1e-4 --interp log --diffuse 2 --void_grade 1e-3 --precond dual --ref_lam_over_mu 10 --tr_metric green"
for d in "0.05 0.5" "0.2 2" "0.5 5"; do set -- $d; echo "--delta0 $1 --delta_max $2"; done \
  | xargs -P 3 -I{} sh -c "OMP_NUM_THREADS=2 JAX_PLATFORMS=cpu MPLBACKEND=Agg $PY stefanus_diffuse.py $C {} 2>&1 \
    | grep '^inc\|TR its, \|failed' | sed 's/^/{}  /'" > exp_data/log_tr_radius.txt

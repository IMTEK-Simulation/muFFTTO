#!/bin/bash
# graded third medium (matrix + inclusion sharp; inside the voids ln mu rises from ln 1e-4 to ln CAP at the wall over
# ~6 w px), 128 px, GPU, simplified material:
#   (A) linear test, w 2, CAP 1e-2 and 1e-3: dual (ref lambda/mu 10) and Green
#   (B) 6 increments to eps 0.03 (buckling), CAP 1e-2, w 2, dual ref 10 + Green-norm trust region (compare _path_sharp)
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
ENV="CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg OMP_NUM_THREADS=2"
MAT="-n 128 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --incl_contrast 10 --lam_over_mu 100 --third_medium 1e-4 --device gpu --interp log --diffuse 2"
{
for CAP in 1e-2 1e-3; do
  echo "--void_grade $CAP --precond dual --ref_lam_over_mu 10 -inc 1 --eps_max 0.0005 --linear_test"
  echo "--void_grade $CAP --precond green -inc 1 --eps_max 0.0005 --linear_test"
done
echo "--void_grade 1e-2 --precond dual --ref_lam_over_mu 10 --tr_metric green -inc 6 --eps_max 0.03 --frames --tag _path_grade1e-2"
} | xargs -P 5 -I{} sh -c "env $ENV timeout 4h $PY stefanus_diffuse.py $MAT {} 2>&1 | grep '^inc\|CG iterations\|TR its, \|Error\|failed' \
    | sed 's/^/{}  /'" >> exp_data/log_void_grade.txt

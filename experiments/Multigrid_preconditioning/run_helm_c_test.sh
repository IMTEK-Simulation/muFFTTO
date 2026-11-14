#!/usr/bin/env bash
# Why does the Helmholtz H_0 help? Smoothing strength HELM_C = 1/16 (matched to the measured 5x curvature growth over
# plane waves) and 4 (stronger) vs the default 1; 32^2 and 64^2, otherwise as run_design_opt_test.sh.
cd "$(dirname "$0")"
PYTHON="${PYTHON:-$HOME/micromamba/envs/muffto-gpu/bin/python}"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONWARNINGS="ignore::RuntimeWarning" SHOW_PLOTS=0 SOFT_EXP=4 \
       PF_NORMALIZED=1 PF_WEIGHT=0.2 CG_MODE=energy CG_DELAY=0 GTOL=0 MAXITER=2000 INIT_FROM=$PWD/data/CH_start_bi_s0.nc
for N in 32 64; do for C in ${CS:-0.0625 4}; do echo "$N $C"; done; done | xargs -P 8 -L 1 bash -c \
  'NPIX=$1 ETA=$(python3 -c "print(4 / $1)") DESIGN_OPT=lbfgs_helm HELM_C=$2 RUN_SUFFIX=_dopt_lbfgs_helm_c$2 '"$PYTHON"' -u example_2D_elasticity_TO.py Green_Jacobi > logs/dopt_N$1_lbfgs_helm_c$2.log 2>&1' _
echo done

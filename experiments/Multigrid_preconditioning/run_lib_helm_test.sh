#!/usr/bin/env bash
# library port check: NuMPI l_bfgs_bounded(precond=Helmholtz) via solvers_nonlinear.minimize vs the local copy
cd "$(dirname "$0")"
PYTHON="${PYTHON:-$HOME/micromamba/envs/muffto-gpu/bin/python}"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONWARNINGS="ignore::RuntimeWarning" SHOW_PLOTS=0 SOFT_EXP=4 \
       PF_NORMALIZED=1 PF_WEIGHT=0.2 CG_MODE=energy CG_DELAY=0 GTOL=0 MAXITER=2000 INIT_FROM=$PWD/data/CH_start_bi_s0.nc
for N in 32 64; do for C in 1 4; do echo "$N $C"; done; done | xargs -P 4 -L 1 bash -c \
  'NPIX=$1 ETA=$(python3 -c "print(4 / $1)") DESIGN_OPT=lbfgs_lib_helm HELM_C=$2 RUN_SUFFIX=_dopt_lib_helm_c$2 '"$PYTHON"' -u example_2D_elasticity_TO.py Green_Jacobi > logs/dopt_N$1_lib_helm_c$2.log 2>&1' _
echo done

#!/usr/bin/env bash
# Design optimizers for TO: NuMPI L-BFGS vs local L-BFGS with Helmholtz H_0, projected PR+ NCG (plain / Helmholtz).
# 32^2 and 64^2, Cahn-Hilliard start bi_s0, NEMO production settings (eta = 4h, contrast 1e4, pfw 0.2, energy CG with
# warm starts), CG_DELAY=0, mechanics preconditioner Green-Jacobi, xtol stop (GTOL=0), MAXITER 2000.
#   systemd-run --user --unit=mg-dopt -p MemoryMax=12G -p MemorySwapMax=0 --working-directory=$PWD bash run_design_opt_test.sh
cd "$(dirname "$0")"
PYTHON="${PYTHON:-$HOME/micromamba/envs/muffto-gpu/bin/python}"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONWARNINGS="ignore::RuntimeWarning" SHOW_PLOTS=0 SOFT_EXP=4 \
       PF_NORMALIZED=1 PF_WEIGHT=0.2 CG_MODE=energy CG_DELAY=0 GTOL=0 MAXITER=2000 INIT_FROM=$PWD/data/CH_start_bi_s0.nc
mkdir -p logs
for N in 32 64; do for V in lbfgs lbfgs_helm ncg ncg_helm; do echo "$N $V"; done; done | xargs -P 8 -L 1 bash -c \
  'NPIX=$1 ETA=$(python3 -c "print(4 / $1)") DESIGN_OPT=$2 RUN_SUFFIX=_dopt_$2 '"$PYTHON"' -u example_2D_elasticity_TO.py Green_Jacobi > logs/dopt_N$1_$2.log 2>&1' _
echo done

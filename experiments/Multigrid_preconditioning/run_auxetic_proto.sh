#!/usr/bin/env bash
# Auxetic target prototype (2026-11-14, user: reduce the target stiffness to get a meaningfully negative Poisson
# ratio): 2D TO 128^2, CH start bi_s11, eta 2h, Helmholtz c 4, contrast 1e3, MG V(2,2);
# TARGET_NU in {-0.5, -0.7} x TARGET_SCALE in {1, 0.3, 0.1} (G* = 0.15 / 0.045 / 0.015 E_0).
cd "$(dirname "$0")"
PYTHON="${PYTHON:-$HOME/micromamba/envs/muffto-gpu/bin/python}"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONWARNINGS="ignore::RuntimeWarning" SHOW_PLOTS=0 SOFT_EXP=3 \
       PF_NORMALIZED=1 PF_WEIGHT=0.2 CG_MODE=energy CG_DELAY=0 GTOL=0 MAXITER=2000 DESIGN_OPT=lbfgs_helm HELM_C=4 \
       INIT_FROM=$PWD/data/CH_starts_new/CH_start_bi_s11.nc NPIX=${NPIX:-128}
export ETA=$(python3 -c "print(2 / $NPIX)")
mkdir -p logs/auxetic
for NU in -0.5 -0.7; do for S in 1 0.3 0.1; do echo "$NU $S"; done; done | xargs -P 6 -L 1 bash -c \
  'TARGET_NU=$1 TARGET_SCALE=$2 RUN_SUFFIX=_aux_nu$1_s$2 '"$PYTHON"' -u example_2D_elasticity_TO.py MG > logs/auxetic/N${NPIX}_nu$1_s$2.log 2>&1' _
echo done

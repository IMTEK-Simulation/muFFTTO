#!/usr/bin/env bash
# Helmholtz H_0 at smaller interface widths: eta = 2h and 1h, 32^2 and 64^2, CH start, production settings,
# Green-Jacobi mechanics: plain L-BFGS, Helmholtz c = 4 (best fixed c at 4h), Helmholtz with the best physical
# length of the 4h sweep, l = sqrt(c) eta = 8h  ->  c = 16 (2h), 64 (1h).
#   systemd-run --user --unit=mg-helmeta -p MemoryMax=14G -p MemorySwapMax=0 --working-directory=$PWD bash run_helm_eta_test.sh
cd "$(dirname "$0")"
PYTHON="${PYTHON:-$HOME/micromamba/envs/muffto-gpu/bin/python}"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONWARNINGS="ignore::RuntimeWarning" SHOW_PLOTS=0 SOFT_EXP=4 \
       PF_NORMALIZED=1 PF_WEIGHT=0.2 CG_MODE=energy CG_DELAY=0 GTOL=0 MAXITER=2000 INIT_FROM=$PWD/data/CH_start_bi_s0.nc
{
for N in 64 32; do
  echo "$N 2 plain 0"; echo "$N 2 helm 4"; echo "$N 2 helm 16"
  echo "$N 1 plain 0"; echo "$N 1 helm 4"; echo "$N 1 helm 64"
done
} | xargs -P 8 -L 1 bash -c '
  N=$1; E=$2; V=$3; C=$4
  if [ $V = plain ]; then OPT=lbfgs; TAG=plain; else OPT=lbfgs_helm; TAG=helm_c$C; fi
  NPIX=$N ETA=$(python3 -c "print($E / $N)") DESIGN_OPT=$OPT HELM_C=${C/#0/1} RUN_SUFFIX=_dopt_eta${E}h_$TAG \
  '"$PYTHON"' -u example_2D_elasticity_TO.py Green_Jacobi > logs/dopt_eta${E}h_N${N}_${TAG}.log 2>&1' _
echo done

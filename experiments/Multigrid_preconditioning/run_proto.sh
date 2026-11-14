#!/usr/bin/env bash
# Prototype the settings for the large runs (2026-11-14; contrast 1e3 = SOFT_EXP 3 by default, user): 2D TO 128^2, new CH start bi_s11, MG mechanics
# (ScaledMaterial hierarchy) + Helmholtz L-BFGS; eta in {2h, 4h} x c in {1, 4, 16} x V(d,d) with d in {2, 1}.
#   systemd-run --user --unit=mg-proto -p MemoryMax=24G -p MemorySwapMax=0 --working-directory=$PWD bash run_proto.sh
cd "$(dirname "$0")"
PYTHON="${PYTHON:-$HOME/micromamba/envs/muffto-gpu/bin/python}"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONWARNINGS="ignore::RuntimeWarning" SHOW_PLOTS=0 SOFT_EXP=${SOFT_EXP:-3} \
       PF_NORMALIZED=1 PF_WEIGHT=0.2 CG_MODE=energy CG_DELAY=0 GTOL=0 MAXITER=2000 DESIGN_OPT=lbfgs_helm \
       INIT_FROM=$PWD/data/CH_starts_new/CH_start_bi_s11.nc NPIX=${NPIX:-128}
mkdir -p logs/proto/soft${SOFT_EXP:-3}
for E in 2 4; do for C in 1 4 16; do for D in 2 1; do echo "$E $C $D"; done; done; done | xargs -P 12 -L 1 bash -c \
  'ETA=$(python3 -c "print($1 / $NPIX)") HELM_C=$2 MG_DEGREE=$3 RUN_SUFFIX=_proto_eta$1h_c$2_V$3 '"$PYTHON"' -u example_2D_elasticity_TO.py MG > logs/proto/soft${SOFT_EXP}/N${NPIX}_eta$1h_c$2_V$3.log 2>&1' _
echo done

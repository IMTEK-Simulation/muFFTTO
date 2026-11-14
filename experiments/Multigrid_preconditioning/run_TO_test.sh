#!/usr/bin/env bash
# TO with multigrid vs the FFT preconditioners: 64^2, NEMO production settings (sbatch_TO2D_start_gpu.sh: contrast 1e4,
# PF_NORMALIZED=1, PF_WEIGHT=0.2, energy CG with warm starts, eta = 4h), CG_DELAY=0, 200 L-BFGS iterations.
#   systemd-run --user --unit=mg-TO -p MemoryMax=12G -p MemorySwapMax=0 --working-directory=$PWD bash run_TO_test.sh
cd "$(dirname "$0")"
PYTHON="${PYTHON:-$HOME/micromamba/envs/muffto-gpu/bin/python}"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONWARNINGS="ignore::RuntimeWarning" SHOW_PLOTS=0 NPIX=${NPIX:-64}
export ETA=$(python3 -c "print(4 / $NPIX)") SOFT_EXP=4 PF_NORMALIZED=1 PF_WEIGHT=0.2 CG_MODE=energy CG_DELAY=0 GTOL=0
export MAXITER=${MAXITER:-200} RUN_SUFFIX=_mgstudy
mkdir -p logs
printf '%s\n' MG Green_Jacobi dual Green | xargs -P 4 -I{} bash -c \
  "$PYTHON -u example_2D_elasticity_TO.py {} > logs/TO_N${NPIX}_{}.log 2>&1"
echo done

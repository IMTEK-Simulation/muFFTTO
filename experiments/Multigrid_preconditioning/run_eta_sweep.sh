#!/usr/bin/env bash
# eta sweep at 256^2 on the phase-field TO design: Green, GJ, dual, MG-PCG (no two-grid cycles)
#   systemd-run --user --unit=mg-eta -p MemoryMax=16G -p MemorySwapMax=0 --working-directory=$PWD bash run_eta_sweep.sh
set -u
cd "$(dirname "$0")"
PYTHON="${PYTHON:-$HOME/micromamba/envs/muffto-gpu/bin/python}"
export OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
mkdir -p logs
printf '%s\n' 0 0.25 0.5 1 1.5 2 3 4 | xargs -P 8 -I{} bash -c \
  "$PYTHON phasefield_twogrid.py --eta {} --cycles --methods green green_jacobi dual mg --tag _sweep > logs/eta_sweep_{}.log 2>&1"
echo done

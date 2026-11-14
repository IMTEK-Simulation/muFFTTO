#!/usr/bin/env bash
# Stage 1 of the multigrid study: 64^2, P1-P3, paper contrasts (+ P3 at 1e-6), all combinations.
#   systemd-run --user --unit=mg-stage1 -p MemoryMax=12G -p MemorySwapMax=0 --working-directory=$PWD bash run_stage1.sh
set -u
cd "$(dirname "$0")"
PYTHON="${PYTHON:-$HOME/micromamba/envs/muffto-gpu/bin/python}"
export OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
N="${N:-64}"
METHODS="green green_jacobi dual mg mgS-green mgS-green_jacobi mgS-dual bnn-green_jacobi bnn3-green bnn3-green_jacobi bnn3-dual add-green_jacobi"
mkdir -p logs
{
for k in -3 -2 -1 1 2 3; do echo "P1 $k"; echo "P2 $k"; done
for k in -1 -2 -3 -6; do echo "P3 $k"; done
} | xargs -P 8 -L 1 bash -c '$0 run_case.py --problem $1 --kappa-exp $2 --grid '"$N"' --methods '"$METHODS"' > logs/stage1_$1_N'"$N"'_k$2.log 2>&1' "$PYTHON"
echo done

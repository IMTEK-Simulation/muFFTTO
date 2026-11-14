#!/usr/bin/env bash
# mesh study 64/128/256 on the phase-field TO design (256^2 design block-averaged for 64, 128), eta/h 0 ... 8:
# fixed eta/h along rows, fixed physical eta along the diagonal (eta/h 2 @ 64 = 4 @ 128 = 8 @ 256)
#   systemd-run --user --unit=mg-mesh -p MemoryMax=16G -p MemorySwapMax=0 --working-directory=$PWD bash run_mesh_study.sh
set -u
cd "$(dirname "$0")"
PYTHON="${PYTHON:-$HOME/micromamba/envs/muffto-gpu/bin/python}"
export OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
mkdir -p logs
{
for eta in 0 0.5 1 2 4 8; do echo "256 $eta"; done          # slow ones first
for N in 128 64; do for eta in 0 0.5 1 2 4 8; do echo "$N $eta"; done; done
} | xargs -P 8 -L 1 bash -c \
  '$0 phasefield_twogrid.py --grid $1 --eta $2 --cycles --methods green green_jacobi dual mg --tag _mesh > logs/mesh_N$1_eta$2.log 2>&1' "$PYTHON"
echo done

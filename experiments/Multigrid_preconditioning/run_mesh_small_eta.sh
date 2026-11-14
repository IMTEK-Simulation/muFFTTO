#!/usr/bin/env bash
# mesh study addendum: eta/h 0.1 and 0.25 at 64^2 and 128^2 (256^2 eta 0.25 from the eta sweep, same geometry)
set -u
cd "$(dirname "$0")"
PYTHON="${PYTHON:-$HOME/micromamba/envs/muffto-gpu/bin/python}"
export OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
printf '%s\n' "128 0.1" "128 0.25" "64 0.1" "64 0.25" | xargs -P 4 -L 1 bash -c \
  '$0 phasefield_twogrid.py --grid $1 --eta $2 --cycles --methods green green_jacobi dual mg --tag _mesh > logs/mesh_N$1_eta$2.log 2>&1' "$PYTHON"
echo done

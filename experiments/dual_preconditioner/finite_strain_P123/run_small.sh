#!/bin/bash
# P1, P2 at kappa = 1e-3 and 1e3, P3 at 1e-3; Green / Green_Jacobi / dual; grids from GRIDS (default 32 64)
#   ./run_small.sh [-j jobs]       -> logs/<geometry>_N<n>_kappa1e<k>_<prec>.log, data/*.npz
cd "$(dirname "$0")"
JOBS=${2:-8}
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
for n in ${GRIDS:-32 64}; do
  for p in Green Green_Jacobi dual; do
    for gk in "P1 -3" "P1 3" "P2 -3" "P2 3" "P3 -3"; do echo "$gk $n $p"; done
  done
done | xargs -P "$JOBS" -L 1 bash -c '
  log="logs/$0_N$2_kappa1e$(printf %+d $1)_$3.log"
  ~/micromamba/envs/muffto-gpu/bin/python -u finite_strain_P123.py -g $0 -k $1 -n $2 -p $3 > "$log" 2>&1
  tail -n 1 "$log"'

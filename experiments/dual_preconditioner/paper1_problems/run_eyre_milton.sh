#!/usr/bin/env bash
# Eyre-Milton vs Green / dual on P1-P3 (conductivity), the paper's contrasts on 256^2.
#   ./run_eyre_milton.sh [-j parallel jobs, default 4]   [PYTHON=... ./run_eyre_milton.sh]
set -u
cd "$(dirname "$0")"
JOBS=4
[[ "${1:-}" == "-j" ]] && JOBS="$2"
PYTHON="${PYTHON:-$HOME/micromamba/envs/muffto-gpu/bin/python}"
export OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
mkdir -p logs_eyre_milton
{
for k in -1 -2 -3 1 2 3; do for prob in P1 P2; do echo "$prob $k"; done; done
for k in -1 -2 -3; do echo "P3 $k"; done
} | xargs -P "$JOBS" -L 1 bash -c \
    "$PYTHON run_eyre_milton.py --problem \$0 --kappa-exp \$1 > logs_eyre_milton/\$0_kappa1e\$1.log 2>&1; tail -1 logs_eyre_milton/\$0_kappa1e\$1.log"

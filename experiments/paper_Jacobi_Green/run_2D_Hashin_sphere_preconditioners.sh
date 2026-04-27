#!/usr/bin/env bash
# Run exp_2D_Hashin_sphere_error_of_preconditioners.py
# for N=128, CG tolerances 1e0 .. 1e-10, all three preconditioners.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXP="$SCRIPT_DIR/exp_2D_Hashin_sphere_error_of_preconditioners.py"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

N=64
PRECS=(Green Green_Jacobi)
TOLS=(  0 1 2 3 4 5 6 7 8 9 10 11 12 ) #

for prec in "${PRECS[@]}"; do
    for tol in "${TOLS[@]}"; do
        echo "=== prec=${prec}  n=${N}  cg_tol=1e-${tol} ==="
        python3 "$EXP" -n "$N" -p "$prec" -cg_tol "$tol"
    done
done

echo "All runs finished."

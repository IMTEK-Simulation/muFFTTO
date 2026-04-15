#!/usr/bin/env bash
set -euo pipefail

# Get the directory where the script is located to ensure relative paths work
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

#etas=( 0.05)
weights=(1.0 )
N=(64)
etas=$(echo "3 *sqrt(2) / $N" | bc -l)

for eta in "${etas[@]}"; do
  for w in "${weights[@]}"; do
    # Corrected: Use '0' instead of '$(0)' to avoid "command not found" error
    for start in 0; do
      stop=$((start + 1000))
      echo "Running with eta=$eta w=$w start=$start stop=$stop"
      
      # Use full path to the python script for better portability
      mpirun -np 8 python "$SCRIPT_DIR/exp_paper_TO_exp_1_square_interphase_length.py" \
        -n "$N" \
        -eta "$eta" \
        -w "$w" \
        -start "$start" \
        -stop "$stop"\
        -cg_tol 8
    done
  done
done

echo "All runs completed."
```
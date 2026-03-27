#!/usr/bin/env bash
set -euo pipefail

# Get the directory where the script is located to ensure relative paths work
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

etas=( 0.01 0.02 0.05 0.005)
weights=(0.1 )

for eta in "${etas[@]}"; do
  for w in "${weights[@]}"; do
    # Corrected: Use '0' instead of '$(0)' to avoid "command not found" error
    for start in 0; do
      stop=$((start + 1000))
      echo "Running with eta=$eta w=$w start=$start stop=$stop"
      
      # Use full path to the python script for better portability
      mpirun -np 4 python "$SCRIPT_DIR/exp_paper_TO_exp_2_square.py" \
        -n 64 \
        -eta "$eta" \
        -w "$w" \
        -start "$start" \
        -stop "$stop"
    done
  done
done

echo "All runs completed."
```
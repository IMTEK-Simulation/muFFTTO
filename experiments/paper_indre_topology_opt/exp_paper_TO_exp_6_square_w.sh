#!/usr/bin/env bash
set -euo pipefail

# Get the directory where the script is located to ensure relative paths work
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Define the list of weight values
w_values=(0.1 0.5 1.0 1.5 2.0 5.0 10.0 20.0)   # 200.0 300.0 400.0 500.0 1000.0)

# Other parameters
N=36
ETA=0.01
POISSON=-0.5
NP=12 # Number of processes for mpirun
ITER_STEP=30 # Number of iterations per weight value

current_start=0
for w in "${w_values[@]}"; do
    current_stop=$((current_start + ITER_STEP))
    
    echo "------------------------------------------------"
    echo "Running exp_paper_TO_exp_6_square.py with w=$w (Iterations $current_start to $current_stop)"
    echo "------------------------------------------------"
    
    mpirun -np "$NP" python "$SCRIPT_DIR/exp_paper_TO_exp_6_square.py" \
        -n "$N" \
        -eta "$ETA" \
        -w "$w" \
        -poisson "$POISSON" \
        -start "$current_start" \
        -stop "$current_stop" \
        -s 10 \
        -r
    current_start=$current_stop
done

echo "All runs completed."

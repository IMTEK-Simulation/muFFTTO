#!/usr/bin/env bash
set -euo pipefail

# Get the directory where the script is located to ensure relative paths work
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Define the list of Poisson values
poissons=(0.0 0.1 0.2 0.3 0.4)

# Other parameters
N=16
ETA=0.01
WEIGHT=5.0
NP=4 # Number of processes for mpirun

for poisson in "${poissons[@]}"; do
    echo "------------------------------------------------"
    echo "Running exp_paper_TO_exp_5_square.py with poisson=$poisson"
    echo "------------------------------------------------"
    
    mpirun -np "$NP" python "$SCRIPT_DIR/exp_paper_TO_exp_5_square.py" \
        -n "$N" \
        -eta "$ETA" \
        -w "$WEIGHT" \
        -poisson "$poisson"\
        -s 10

done

echo "All runs completed."

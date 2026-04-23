#!/bin/bash

# Configuration: Define number of cores for each grid size
declare -A CORE_MAP
CORE_MAP[16]=2
CORE_MAP[32]=4
CORE_MAP[64]=8
CORE_MAP[128]=12
CORE_MAP[256]=16

# Paths
SCRIPT_NAME="run_grid_convergence_study.py"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Grid sizes, Preconditioners, and Soft Phase Exponents
GRID_SIZES=(128 )
PRECONDITIONERS=( "Green_Jacobi") #"Green"
SOFT_EXPONENTS=(0) # 3 5

echo "--------------------------------------------------------"
echo "Starting Parallel Grid Convergence Study"
echo "Script: $SCRIPT_NAME"
echo "Directory: $SCRIPT_DIR"
echo "--------------------------------------------------------"

for n in "${GRID_SIZES[@]}"; do
    cores=${CORE_MAP[$n]:-2}
    
    for p_type in "${PRECONDITIONERS[@]}"; do
        for soft in "${SOFT_EXPONENTS[@]}"; do
            echo ">>> RUNNING: N=$n, Preconditioner=$p_type, Soft=$soft on $cores cores"
            
            export EXP_N=$n
            export EXP_P=$p_type
            export EXP_SOFT=$soft
            
            mpirun -n $cores python3 "$SCRIPT_DIR/$SCRIPT_NAME"
        done
    done
done

echo "--------------------------------------------------------"
echo "Study completed."
echo "--------------------------------------------------------"

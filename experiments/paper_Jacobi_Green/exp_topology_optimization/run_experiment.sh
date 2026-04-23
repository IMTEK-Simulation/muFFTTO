#!/bin/bash

# Configuration
SCRIPT_NAME="run_experiment_template.py"
NUM_PROCS=2

# Get the directory where this bash script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "--------------------------------------------------------"
echo "Starting Topology Optimization Experiment"
echo "Script: $SCRIPT_NAME"
echo "Parallel processes: $NUM_PROCS"
echo "Directory: $SCRIPT_DIR"
echo "--------------------------------------------------------"

# Run with mpirun using the absolute path to the python script
mpirun -n $NUM_PROCS python3 "$SCRIPT_DIR/$SCRIPT_NAME"

echo "--------------------------------------------------------"
echo "Experiment finished."
echo "--------------------------------------------------------"

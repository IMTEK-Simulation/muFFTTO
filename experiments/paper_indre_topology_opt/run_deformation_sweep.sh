#!/bin/bash

# Script to run exp_paper_TO_compute_deformation_for_plots.py with different poison_target values
# from -0.5 to 0.4 with step 0.1

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to muFFTTO root directory (2 levels up from experiments/paper_indre_topology_opt/)
MUFFFTO_ROOT="$SCRIPT_DIR/../.."
cd "$MUFFFTO_ROOT"

# Add muFFTTO root to PYTHONPATH so the module can be imported
export PYTHONPATH="$PWD:$PYTHONPATH"

SCRIPT_PATH="experiments/paper_indre_topology_opt/exp_paper_TO_compute_deformation_for_plots.py"

# Loop through grid types
for grid in hex square
do
    echo "##########################################"
    echo "# Starting grid_type = $grid"
    echo "##########################################"

    # Loop through weight values
    for weight in 10.0 20.0
    do
        echo "########################################## "
        echo "Starting grid_type = $grid, weight = $weight"
        echo "########################################## "

        # Loop through poison_target values from -0.5 to 0.4 with step 0.1
        for value in -0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4
        do
            echo "=========================================="
            echo "Running with grid_type = $grid, weight = $weight, poison_target = $value"
            echo "=========================================="

            # First run: Compute with MPI (12 processes)
            echo "Step 1: Computing with MPI..."
            mpirun -n 12 python $SCRIPT_PATH --grid_type $grid --weight $weight --poison_target $value

            if [ $? -ne 0 ]; then
                echo "ERROR: MPI computation failed for grid_type = $grid, weight = $weight, poison_target = $value"
                exit 1
            fi

            # Second run: Generate movie without MPI
            echo "Step 2: Generating movie..."
            python $SCRIPT_PATH --grid_type $grid --weight $weight --poison_target $value

            if [ $? -ne 0 ]; then
                echo "ERROR: Movie generation failed for grid_type = $grid, weight = $weight, poison_target = $value"
                exit 1
            fi

            echo "Completed grid_type = $grid, weight = $weight, poison_target = $value"
            echo ""
        done

        echo "########################################## "
        echo "Finished all poison_target values for grid_type = $grid, weight = $weight"
        echo "########################################## "
        echo ""
    done

    echo "##########################################"
    echo "# Finished all weights for grid_type = $grid"
    echo "##########################################"
    echo ""
done

echo "=========================================="
echo "All computations completed successfully!"
echo "=========================================="

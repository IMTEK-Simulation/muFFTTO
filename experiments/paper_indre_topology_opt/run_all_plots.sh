#!/bin/bash

# Define the directory where the scripts are located
SCRIPT_DIR="./"

# List of scripts to run
SCRIPTS=(
    "exp_paper_TO_exp_2_hexa_w_dep_plots.py"
    "exp_paper_TO_exp_2_square_w_dep_plots.py"
    "exp_paper_TO_exp_3_hexa_w_dep_plots.py"
    "exp_paper_TO_exp_3_square_w_dep_plots.py"
    "exp_paper_TO_exp_4_hexa_w_dep_plots.py"
    "exp_paper_TO_exp_4_square_w_dep_plots.py"
)

# Run each script
for SCRIPT in "${SCRIPTS[@]}"; do
    echo "Running $SCRIPT..."
    python3 "$SCRIPT_DIR/$SCRIPT"
done

echo "All scripts have been executed."

#!/bin/bash
#SBATCH --job-name=grid_conv
#SBATCH --nodes=01
#SBATCH --ntasks-per-node=16
#SBATCH --time=12:00:00
#SBATCH --mem=32gb

echo "Starting Parallel Grid Convergence Study on Cluster"

# Add muFFTTO to Python path - matching the actual directory on cluster
# We add the root directory where 'muFFTTO' package is located
export PYTHONPATH="/work/classic/fr_ml1145-jacobi_paper:$PYTHONPATH"

# Change to the correct directory on cluster
cd /work/classic/fr_ml1145-jacobi_paper

# Configuration: Define number of cores for each grid size
declare -A CORE_MAP
CORE_MAP[16]=2
CORE_MAP[32]=4
CORE_MAP[64]=8
CORE_MAP[128]=12
CORE_MAP[256]=16

# Paths - These should be relative to the cd'd directory above if possible, 
# but I will use the structure from the provided local script.
# Assuming we are in /work/classic/fr_ml1145-jacobi_paper/
EXP_DIR="./muFFTTO/experiments/paper_Jacobi_Green/exp_topology_optimization/grid_convergence_test"
SCRIPT_NAME="run_grid_convergence_study.py"
CONTAINER="/work/classic/fr_ml1145-jacobi_paper/2026-04-23-nemo2-mugrid-0.105.2.sif"

# Grid sizes, Preconditioners, and Soft Phase Exponents
GRID_SIZES=(16 32 64 128 256)
PRECONDITIONERS=("Green" "Green_Jacobi")
SOFT_EXPONENTS=(0)

for n in "${GRID_SIZES[@]}"; do
    cores=${CORE_MAP[$n]:-2}
    
    for p_type in "${PRECONDITIONERS[@]}"; do
        for soft in "${SOFT_EXPONENTS[@]}"; do
            echo ">>> RUNNING: N=$n, Preconditioner=$p_type, Soft=$soft on $cores cores"
            
            export EXP_N=$n
            export EXP_P=$p_type
            export EXP_SOFT=$soft
            
            # Using srun with apptainer as per template
            srun --ntasks=$cores apptainer exec $CONTAINER \
                python3 "$EXP_DIR/$SCRIPT_NAME"
        done
    done
done

echo "--------------------------------------------------------"
echo "Study completed."
echo "--------------------------------------------------------"

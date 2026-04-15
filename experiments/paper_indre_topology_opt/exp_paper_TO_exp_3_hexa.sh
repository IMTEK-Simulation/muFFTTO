#!/bin/bash
#SBATCH --job-name=neg_to 
#SBATCH --nodes=01
#SBATCH --ntasks-per-node=32
#SBATCH --time=4:00:00
#SBATCH --mem=16gb
#SBATCH --array=0-31 #143   # 4 etas * 36 weights = 144 tasks

echo "ID/NAME:    $SLURM_JOB_ID / $SLURM_JOB_NAME"
echo "USER:       $SLURM_JOB_USER"
echo "PARTITION:  $SLURM_JOB_PARTITION"
echo "TASKS:      $SLURM_TASKS_PER_NODE tasks/node x $SLURM_JOB_NUM_NODES nodes = $SLURM_NTASKS tasks"
echo "NODES:      $SLURM_JOB_NODELIST"

# Add muFFTTO to Python path
export PYTHONPATH="/work/classic/fr_ml1145-martin_workspace_01/muFFTTO:$PYTHONPATH"

# Change to the correct directory
cd /work/classic/fr_ml1145-martin_workspace_01

N=64
eta_values=   # (0.01 0.02)
w_values=(0.1  0.5  1.0   1.5   2.0   5.0   10.0 20.0 30.0 50.0  100.0  200.0 300.0 400.0 500.0 1000.)

# Calculate indices for eta and weight based on the array task ID
# This maps 0-119 to all combinations of the two arrays
eta_idx=$(( SLURM_ARRAY_TASK_ID / 36 ))
w_idx=$(( SLURM_ARRAY_TASK_ID % 36 ))

eta=${eta_values[$eta_idx]}
w=${w_values[$w_idx]}

echo "Running task $SLURM_ARRAY_TASK_ID: eta=$eta, w=$w"

# Run the command using calculated eta and weight
srun apptainer exec ./mufft-2025-12-15-fixed-memory-leak.sif \
    python3 ./muFFTTO/experiments/paper_indre_topology_opt/exp_paper_TO_exp_3_hexa.py \
    -n $N \
    -p "Green_Jacobi" \
    -cg_tol 8 \
    -eta "$eta" \
    -w "$w" \
    -stop 4000 \
    -soft 5
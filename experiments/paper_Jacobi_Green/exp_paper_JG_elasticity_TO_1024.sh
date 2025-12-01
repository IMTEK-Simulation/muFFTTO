#! /bin/bash

NUM_RUNS=2

# List of preconditioner types (strings)
PRECONDS=("Green" "Jacobi" "Green_Jacobi")


for i in $(seq 1 $NUM_RUNS); do
    for prec in "${PRECONDS[@]}"; do
          NP=10
          echo "$i"
          echo "Run iteration $i with preconditioner $prec,  using -np $NP..."
          mpirun -np $NP python ./exp_paper_JG_2D_elasticity_TO_1024.py -n 128 -it $i -p "$prec"

    done
done

echo "All runs completed."

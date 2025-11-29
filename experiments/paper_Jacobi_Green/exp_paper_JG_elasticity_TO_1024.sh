#! /bin/bash

NUM_RUNS=1200

# List of preconditioner types (strings)
PRECONDS=("Green" "Jacobi" "Green_Jacobi")


for i in $(seq 153 $NUM_RUNS); do
    for prec in "${PRECONDS[@]}"; do
          NP=6
          echo "$i"
          echo "Run iteration $i with preconditioner $prec,  using -np $NP..."
          mpirun -np $NP python ./exp_paper_JG_2D_elasticity_TO_1024.py -n 64 -it $i -p "$prec"

    done
done

echo "All runs completed."

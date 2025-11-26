#! /bin/bash

NUM_RUNS=4

# List of preconditioner types (strings)
PRECONDS=("Green"   "Green_Jacobi")


for i in $(seq 3 $NUM_RUNS); do
    for prec in "${PRECONDS[@]}"; do
          if [ $i -gt 5 ]; then
              NP=8
          else
              NP=4
          fi
          echo "Run $i with preconditioner $prec,  using -np $NP..."
          mpirun -np $NP python ./exp_paper_JG_nonlinear_elasticity_JZ.py -n $((2**$i)) -p "$prec"

    done
done

echo "All runs completed."


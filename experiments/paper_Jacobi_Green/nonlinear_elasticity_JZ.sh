#! /bin/bash

NUM_RUNS=5

# List of preconditioner types (strings)
PRECONDS=("Green"   "Green_Jacobi")


for i in $(seq 5 $NUM_RUNS); do
  for exp in 3 4 5; do #
    for prec in "${PRECONDS[@]}"; do
          if [ $i -gt 5 ]; then
              NP=12
          else
              NP=4
          fi
          echo "Run $i with preconditioner $prec,  using -np $NP..."
          #mpirun -np $NP python ./exp_paper_JG_nonlinear_elasticity_JZ.py -n $((2**$i)) -exp $exp -p "$prec"
          mpirun -np $NP python ./exp_paper_JG_nonlinear_elasticity_JZ_bubles.py -n $((2**$i)) -exp $exp -p "$prec"
    done
  done
done

echo "All runs completed."


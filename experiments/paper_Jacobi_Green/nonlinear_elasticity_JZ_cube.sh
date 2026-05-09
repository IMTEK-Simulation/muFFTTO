#! /bin/bash

NUM_RUNS=4
# List of preconditioner types (strings)
PRECONDS=("Green" "Green_Jacobi" ) #"Green" "Green_Jacobi"


for i in $(seq 4 $NUM_RUNS); do
  for exp in   2 ; do # #
    for prec in "${PRECONDS[@]}"; do
          if [ $i -gt 4 ]; then
              NP=12
          else
              NP=4
          fi
          echo "Run $i with preconditioner $prec,  using -np $NP..."
          #mpirun -np $NP python ./exp_paper_JG_nonlinear_elasticity_JZ.py -n $((2**$i)) -exp $exp -p "$prec"
          mpirun -np $NP python ./exp_paper_JG_nonlinear_elasticity_JZ_cube.py -n $((2**$i)) -exp $exp -p "$prec"
    done
  done
done

echo "All runs completed."


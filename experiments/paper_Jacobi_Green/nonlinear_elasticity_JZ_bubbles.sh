#! /bin/bash

NUM_RUNS=5
# List of preconditioner types (strings)
PRECONDS=("Green" "Green_Jacobi" ) #"Green" "Green_Jacobi"


for i in $(seq 4 $NUM_RUNS); do
  for exp in 5 ; do # #
    for prec in "${PRECONDS[@]}"; do
          if [ $i -gt 4 ]; then
              NP=8
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


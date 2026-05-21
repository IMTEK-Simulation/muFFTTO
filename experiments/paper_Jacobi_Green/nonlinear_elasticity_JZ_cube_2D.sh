#! /bin/bash

NUM_RUNS=6
# List of preconditioner types (strings)
PRECONDS=( "Green_Jacobi" "Green" ) #"Green" "Green_Jacobi""Green"


for i in $(seq 4 $NUM_RUNS); do
  for exp in    3 ; do # #
    for prec in "${PRECONDS[@]}"; do
          if [ $i -gt 5 ]; then
              NP=8
          else
              NP=4
          fi
          echo "Run $i with preconditioner $prec,  using -np $NP..."
          #mpirun -np $NP python ./exp_paper_JG_nonlinear_elasticity_JZ.py -n $((2**$i)) -exp $exp -p "$prec"
          mpirun -np $NP python ./exp_paper_JG_nonlinear_elasticity_JZ_cube_2D.py -n $((2**$i)) -exp $exp -p "$prec"
    done
  done
done

echo "All runs completed."


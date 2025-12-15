#! /bin/bash
# List of preconditioner types (strings)
PRECONDS=("Green"   "Green_Jacobi")

for start in $(seq 0 11 1000); do
    stop=$((start + 10))
 echo "Running with start=$start stop=$stop"
mpirun -np 4 python ./exp_paper_JG_2D_elasticity_TO_load_init.py -n 16 -p "Green_Jacobi" -start $start -stop $stop

done


echo "All runs completed."


#! /bin/bash

NUM_RUNS=13

# List of preconditioner types (strings)
#PRECONDS=("Green")# "Jacobi"
PRECONDS=("Green" "Green_Jacobi")

# List of numeric p values
PVALUES=(1 )

for i in $(seq 10 $NUM_RUNS); do
    for prec in "${PRECONDS[@]}"; do
        for pv in "${PVALUES[@]}"; do
            if [ $i -gt 7 ]; then
                NP=1
            else
                NP=1
            fi
            echo "Run $i with preconditioner $prec, rho=$pv, using -np $NP..."
            mpirun -np $NP python ./exp_paper_JG_nlinear.py -n $i -p "$prec" -rho "$pv"
        done
    done
done

echo "All runs completed."

#
## List of preconditioner types (strings)
#PRECONDS=( "Green_Jacobi" "Jacobi") #"Green" "Green_Jacobi" "Jacobi"
## List of numeric p values
#PVALUES=(4 ) # 0
#
#for i in $(seq 2 $NUM_RUNS); do
#    for prec in "${PRECONDS[@]}"; do
#        for pv in "${PVALUES[@]}"; do
#            if [ $i -gt 7 ]; then
#                NP=8
#            else
#                NP=4
#            fi
#            echo "Run $i with preconditioner $prec, rho=$pv, using -np $NP..."
#            mpirun -np $NP python ./exp_paper_JG_cos.py -n $i -p "$prec" -rho "$pv"
#        done
#    done
#done
#
#echo "All runs completed."
#! /bin/bash

NUM_RUNS=10

# List of preconditioner types (strings)
PRECONDS=("Green" "Jacobi" "Green_Jacobi")

# List of numeric p values
PVALUES=(1 4 )

for i in $(seq 2 $NUM_RUNS); do
    for prec in "${PRECONDS[@]}"; do
        for pv in "${PVALUES[@]}"; do
            if [ $i -gt 7 ]; then
                NP=10
            else
                NP=4
            fi
            echo "Run $i with preconditioner $prec, rho=$pv, using -np $NP..."
            mpirun -np $NP python ./exp_paper_JG_cos.py -n $i -p "$prec" -rho "$pv"
        done
    done
done

echo "All runs completed."

# List of numeric p values
PVALUES=(0 4 )

for i in $(seq 2 $NUM_RUNS); do
    for prec in "${PRECONDS[@]}"; do
        for pv in "${PVALUES[@]}"; do
            if [ $i -gt 7 ]; then
                NP=10
            else
                NP=4
            fi
            echo "Run $i with preconditioner $prec, rho=$pv, using -np $NP..."
            mpirun -np $NP python ./exp_paper_JG_cos.py -n $i -p "$prec" -rho "$pv"
        done
    done
done

echo "All runs completed."
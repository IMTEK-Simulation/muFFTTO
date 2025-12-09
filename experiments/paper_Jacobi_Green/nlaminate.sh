#! /bin/bash

NUM_RUNS=4
NORMS=("norm_rr") #"norm_rGr"
# List of preconditioner types (strings)
PRECONDS=(  "Jacobi") #"Green" "Green_Jacobi" "Jacobi""Green_Jacobi"
# List of numeric p values
PVALUES=(0 ) # 0

for norm in "${NORMS[@]}"; do
  for i in $(seq 2 $NUM_RUNS); do
      for prec in "${PRECONDS[@]}"; do
          for pv in "${PVALUES[@]}"; do
                for exp in $(seq -16 0); do
                  zero=$(awk "BEGIN{printf \"%.2e\", 10^$exp}")
                  echo "Processing with i = $i"
                  if [ $i -gt 7 ]; then
                      NP=8
                  else
                      NP=4
                  fi
                  echo "Run $i with  -p $prec -rho $pv -norm $norm"
                  mpirun -np $NP python ./exp_paper_JG_cos.py -n $i -p "$prec" -rho "$pv" -nor $norm -jz  $zero
              done
          done
      done
  done
done

echo "All runs completed."q

pica
NUM_RUNS=0

# List of preconditioner types (strings)
#PRECONDS=("Green")# "Jacobi"
PRECONDS=("Green" "Green_Jacobi" "Jacobi")

# List of numeric p values
PVALUES=(1 4)

for pv in "${PVALUES[@]}"; do
  for prec in "${PRECONDS[@]}"; do
    for i in $(seq 2 $NUM_RUNS); do

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


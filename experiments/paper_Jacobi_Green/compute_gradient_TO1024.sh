#! /bin/bash

NUM_RUNS=1000


for i in $(seq 1 $NUM_RUNS); do

          echo "Run iteration $i    using -np 12..."
          mpirun -np 12 python ././exp_paper_JG_2D_elasticity_TO_1024_compute_gradient.py  -n 512 -it $i


done

echo "All runs completed."


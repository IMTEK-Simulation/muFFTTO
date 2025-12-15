#! /bin/bash
# List of preconditioner types (strings)


#!/bin/bash

for exp in $(seq 6 11); do
  nodes=$((2**exp))

  for cores in $(seq 1 12); do
   echo "Running with cores=$cores "
  mpirun -np $cores python ./performance_test_convolution.py -n $nodes

done
done

echo "All runs completed."


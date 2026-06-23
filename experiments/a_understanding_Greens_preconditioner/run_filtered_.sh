#! /bin/bash

# Step 2: run CG solver for each size and filter count
for nb_nodes in 16 32 64 128 256 512 1024; do #
  for nb_filter in $(seq 1 1 ); do
    echo "Run n=$nb_nodes nb_filter=$nb_filter"
    mpirun -np 1 python3 ././filtered_geometry_conductivity_CG_evolution.py -n $nb_nodes -nb_filter $nb_filter -p "Green" -g 'circle_inclusion'
  done
done

echo "All runs completed."


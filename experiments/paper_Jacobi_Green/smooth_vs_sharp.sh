#! /bin/bash


#
#mpirun -np 12 python ././exp_paper_smooth_vs_sharp_interphases_1024.py -r 5 -cg_tol 5



for rats in {2,}; do
  echo "Run cg_tol 5  ratios $rats using -np 12..."
  mpirun -n 12 python ././exp_paper_smooth_vs_sharp_interphases_1024.py -r $rats -cg_tol 5

done

echo "All runs completed."


#! /bin/bash



 mpirun -np 10 python ././exp_paper_smooth_vs_sharp_interphases_1024.py -r 4 -cg_tol 1
#
#NUM_RUNS=3
#
#
#for i in $(seq 1 $NUM_RUNS); do
#  for rats in {2,}; do
#
#          echo "Run cg_tol $i  ratios $rats using -np 10..."
#          mpirun -np 12 python ././exp_paper_smooth_vs_sharp_interphases_1024.py -r $rats -cg_tol $i
#
#  done
#done
#
#echo "All runs completed."
#

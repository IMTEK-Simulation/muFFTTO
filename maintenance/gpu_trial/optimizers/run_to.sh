#!/bin/bash
M=$HOME/PycharmProjects/muFFTTO/exp_data/gpu_trial_scratch/mma
export MAMBA_ROOT_PREFIX=~/micromamba
eval "$(~/.local/bin/micromamba shell hook -s bash)"
micromamba activate muffto-gpu
export MPLBACKEND=Agg JAX_PLATFORMS=cpu
for opt in lbfgs mma; do
  ( cd $M/to_$opt || exit 1; /usr/bin/time -f "%e s wall" -o time.txt python -u to.py < /dev/null > stdout.log 2> stderr.log; echo "rc=$?" > rc.txt ) &
done
wait

#!/bin/bash
export MAMBA_ROOT_PREFIX=~/micromamba
eval "$(~/.local/bin/micromamba shell hook -s bash)"
micromamba activate muffto-gpu
export MPLBACKEND=Agg JAX_PLATFORMS=cpu OMP_NUM_THREADS=2
for d in lbfgs mma lbfgs06 mma06; do
  ( cd $HOME/PycharmProjects/muFFTTO/exp_data/gpu_trial_scratch/mma/v2/$d || exit 1; /usr/bin/time -f "%e s wall" -o time.txt python -u to.py < /dev/null > stdout.log 2> stderr.log; echo "rc=$?" > rc.txt ) &
done
wait

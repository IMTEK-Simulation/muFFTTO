#!/bin/bash
# run_set.sh FOLDER SLIDE... : run the given slides of FOLDER one after the other (time.txt, stdout.log per slide)
export JAX_PLATFORMS=cpu MPLBACKEND=Agg OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_PATH=/home/martin/micromamba/envs/muffto-gpu/targets/x86_64-linux
folder=$1; shift
for slide in "$@"; do
  ( cd "$folder/$slide" && /usr/bin/time -f "%e s wall" -o time.txt ~/micromamba/envs/muffto-gpu/bin/python -u run.py \
      < /dev/null > stdout.log 2> stderr.log )
  echo "$folder/$slide rc=$? $(cat $folder/$slide/time.txt | tail -1) | $(grep -E '^inc' $folder/$slide/stdout.log | tail -1 | cut -c1-60)"
done

#!/bin/bash
# memory of none / Green / dual / MG, one process per configuration, sequential (device memory is measured globally)
# -> exp_data/memory_footprint.jsonl ; then python plot_memory_footprint.py
cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_PATH=~/micromamba/envs/muffto-gpu/targets/x86_64-linux
PY=~/micromamba/envs/muffto-gpu/bin/python
OUT=${OUT:-exp_data/memory_footprint.jsonl}
run() {   # device dim N...
  local dev=$1 d=$2; shift 2
  for N in "$@"; do for P in none Green dual MG; do
    DEVICE=$dev OMP_NUM_THREADS=1 timeout 1800 $PY memory_footprint.py $P $N $d 2>logs/memory_footprint.err | tail -n 1 >> $OUT \
      || echo "{\"precond\": \"$P\", \"N\": $N, \"dim\": $d, \"device\": \"$dev\", \"failed\": true}" >> $OUT
  done; done
}
run gpu 2 512 1024 2048
run gpu 3 32 64 128
# run cpu 2 512 1024 2048
# run cpu 3 32 64 96
echo done

#!/bin/bash
# peak GPU memory of the AT2 script itself (whole run: elastic + damage), 2 load steps, per elastic preconditioner;
# per-process memory sampled by nvidia-smi every 0.1 s -> exp_data/memory_at2.txt
cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_PATH=~/micromamba/envs/muffto-gpu/targets/x86_64-linux
PY=~/micromamba/envs/muffto-gpu/bin/python
OUT=exp_data/memory_at2.txt
: > $OUT
for N in 512 1024 2048; do for P in Green dual MG; do
  DEVICE=gpu MICROSTRUCTURE=polydisperse_pores NPIX=$N LENGTH_SCALE=$(python3 -c "print(2/$N)") \
    GC=$(python3 -c "print(0.032/$N)") LOAD_MAX=0.002 LOAD_STEPS=2 OUTPUT_NC=0 CHECKPOINT_EVERY=0 \
    MG_DEGREE=1 MG_LMAX=gershgorin RUN_SUFFIX=_memtest \
    $PY example_2D_phase_field_fracture_AT2.py $P > logs/memory_at2_N${N}_$P.log 2>&1 &
  pid=$!; max=0
  while kill -0 $pid 2>/dev/null; do
    m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$pid '$1==p{print $2}')
    [ -n "$m" ] && [ "$m" -gt "$max" ] && max=$m
    sleep 0.1
  done
  wait $pid; echo "$N $P $max $?" >> $OUT
done; done
rm -f exp_data/*_memtest_* figures/*_memtest*
echo done

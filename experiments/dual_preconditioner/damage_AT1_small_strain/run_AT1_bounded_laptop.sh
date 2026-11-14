#!/bin/bash
# Laptop prototype of the AT1 study with the BOUNDED AT1 (example_2D_phase_field_fracture_AT1_bounded.py):
# B2 nine pores, same load as sbatch_AT1.sh (LOAD_MAX 0.32, LOAD_STEPS 320), Green and dual.
#   A: fixed l = 1/32 on 32, 64, 128 (CPU) and 256 (GPU, one after the other)  -> l/h = 1, 2, 4, 8
#   B: 128^2, l/h = 0.25, 0.5, 1, 2 (+ 4 from A)                               -> l = LH / 128
# output: exp_data/AT1_N{N}_{P}_polydisperse_pores_l{l}_bounded{.nc,_iterations.csv}; logs/AT1b_*.log
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export MPLBACKEND=Agg MODEL=AT1 MICROSTRUCTURE=polydisperse_pores LOAD_MAX=0.32 LOAD_STEPS=320
mkdir -p logs
cpu() {  # N l P
    OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu DEVICE=cpu NPIX=$1 LENGTH_SCALE=$2 \
        $PY -u example_2D_phase_field_fracture_AT1_bounded.py $3 > logs/AT1b_N$1_$3_l$2.log 2>&1 &
}
for P in Green dual; do
    for N in 32 64 128; do cpu $N 0.03125 $P; done
    for L in 0.001953125 0.00390625 0.0078125 0.015625; do cpu 128 $L $P; done
done
( for P in dual Green; do
    DEVICE=gpu NPIX=256 LENGTH_SCALE=0.03125 $PY -u example_2D_phase_field_fracture_AT1_bounded.py $P \
        > logs/AT1b_N256_${P}_l0.03125.log 2>&1
  done ) &
wait
grep -H "Elapsed" logs/AT1b_*.log

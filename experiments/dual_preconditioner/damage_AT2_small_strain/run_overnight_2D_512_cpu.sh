#!/bin/bash
# overnight 2026-10-31, CPU: 2D AT2 512^2 at fixed l/h = 2, 4 (l = 1/256, 1/128), pores and stiff tough inclusions (10x),
# dual and Green (Green-Jacobi was the slowest in every 2D fixed-l/h case up to 128^2); 4 MPI ranks per run, two lanes.
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu MPLBACKEND=Agg OMP_NUM_THREADS=1 MICROSTRUCTURE=polydisperse_pores NPIX=512 RUN_SUFFIX=_lh
PY=~/micromamba/envs/muffto-gpu/bin/python
MPI="$HOME/micromamba/envs/muffto-gpu/bin/mpirun -n 4 --bind-to none"
run() {  # L INCL P
    tag=$([ -n "$2" ] && echo "_incl$2")_l$1_lh
    echo "$(date +%H:%M:%S) start l=$1 incl=$2 $3"
    env LENGTH_SCALE=$1 ${2:+INCLUSION_STIFFNESS=$2} $MPI $PY -u example_2D_phase_field_fracture_AT2.py $3 \
        > logs/AT2_N512_$3_polydisperse_pores${tag}.log 2>&1
    echo "$(date +%H:%M:%S) done  l=$1 incl=$2 $3 (exit $?)"
}
(run 0.00390625 10 dual; run 0.00390625 10 Green; run 0.0078125 10 dual; run 0.0078125 10 Green) &
(run 0.00390625 "" dual; run 0.00390625 "" Green; run 0.0078125 "" dual; run 0.0078125 "" Green) &
wait

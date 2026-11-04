#!/bin/bash
# 2D: how the smoothness of the damage profile (l/h) changes the grid scaling of Green, Green-Jacobi and dual.
# Full runs on a grid x length-scale matrix: N = 32, 64, 128 and l = 1/8, 1/16, 1/32, 1/64 with l/h = N l >= 1
# (fixed l = smooth-crack line, fixed l/h = sharp-crack line), pores and stiff inclusions (10x), all three
# preconditioners; 12 runs at a time; output suffix _smooth
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export MICROSTRUCTURE=polydisperse_pores MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu RUN_SUFFIX=_smooth
job() {  # N L INCL P
    tag=$([ "$3" != "-" ] && echo "_incl$3")_l$2
    systemd-run --user --scope -p MemoryMax=4G -q env NPIX=$1 LENGTH_SCALE=$2 \
        $([ "$3" != "-" ] && echo "INCLUSION_STIFFNESS=$3") \
        $PY -u example_2D_phase_field_fracture_AT2.py $4 > logs/AT2_N$1_$4_polydisperse_pores${tag}_smooth.log 2>&1
    echo "$(date +%H:%M:%S) done N=$1 l=$2 incl=$3 $4"
}
export -f job; export PY
for N in 128 64 32; do for L in 0.125 0.0625 0.03125 0.015625; do
    awk -v n=$N -v l=$L 'BEGIN{exit !(n*l >= 1)}' || continue
    for INCL in - 10; do for P in Green Green_Jacobi dual; do echo "$N $L $INCL $P"; done; done
done; done | xargs -P 12 -L 1 bash -c 'job $0 $1 $2 $3'

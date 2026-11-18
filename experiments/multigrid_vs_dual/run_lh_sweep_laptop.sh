#!/bin/bash
# Laptop part of the l/h sweep (CPU, one thread per run, 8 runs at a time; the six configurations of one (N, l/h) are
# started next to each other, so their wall times share the machine load): N = 64, 128; l/h = 0.5 1 1.5 2 3 4;
# dual, MG V(1,1), MG V(2,2) x d_min 10, 0. Same problem as sbatch_lh_sweep.sh (Gc = 0.016 l, load to 0.12 in 120
# steps, no NetCDF). Finished runs (log with 'Elapsed time') are skipped.
#   MG_LMAX=gershgorin (default here): lambda_max(D^-1 A) from the rigorous bound 3.523, no power iteration; runs and
#   logs tagged SUFFIX=_lb. The first pass (MG_LMAX=power, MG_POWER=3, no tag) broke down at the 1024^2 cut-through.
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu MPLBACKEND=Agg OMP_NUM_THREADS=1 PYTHONWARNINGS="ignore::RuntimeWarning"
export PY=~/micromamba/envs/muffto-gpu/bin/python MG_LMAX=${MG_LMAX:-gershgorin} SUFFIX=${SUFFIX-_lb}
one() {   # one <N> <lh> <config>
    local N=$1 LH=$2 C=$3 P TAG DG D DT
    case $C in dual*) P=dual; TAG=; DG=2 ;; MG11*) P=MG; TAG=_v11; DG=1 ;; MG22*) P=MG; TAG=_v22; DG=2 ;; esac
    case $C in *_d0) D=0; DT=_d0 ;; *) D=10; DT= ;; esac
    local LOG=logs/AT2_N${N}_${C}_sw_lh${LH}${SUFFIX}.log
    [ -f $LOG ] && grep -q "Elapsed time" $LOG && return
    env MICROSTRUCTURE=polydisperse_pores NPIX=$N LENGTH_SCALE=$($PY -c "print($LH / $N)") \
        GC=$($PY -c "print(0.016 * $LH / $N)") LOAD_MAX=0.12 LOAD_STEPS=120 CHECKPOINT_EVERY=0 OUTPUT_NC=0 \
        MG_POWER=3 MG_DEGREE=$DG CG_DELAY=$D RUN_SUFFIX=_sw_lh${LH}${TAG}${DT}${SUFFIX} \
        $PY -u example_2D_phase_field_fracture_AT2.py $P > $LOG 2>&1
}
export -f one
for N in 64 128; do for LH in 0.5 1 1.5 2 3 4; do for C in dual MG11 MG22 dual_d0 MG11_d0 MG22_d0; do
    echo "$N $LH $C"; done; done; done | xargs -P 8 -L 1 bash -c 'one "$0" "$1" "$2"'

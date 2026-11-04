#!/bin/bash
# check of the 32^2 - 128^2 extrapolation (plot_AT2_contrast_extrapolation_2D.py): full runs at 256^2 for the five
# material settings of the contrast study, all three preconditioners, the sharp (l/h = 2, l = 1/128) and the smooth
# (l = 1/16, l/h = 16) line; 12 runs at a time; output suffix _smooth as the 32^2 - 128^2 runs
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export MICROSTRUCTURE=polydisperse_pores MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu RUN_SUFFIX=_smooth NPIX=256
job() {  # SETTING L P
    case $1 in
        poresk3)  envs="HOLES=1 HOLE_STIFFNESS=1e-3 K_RESIDUAL=1e-3"; tag=_holes0.001_l$2_k0.001 ;;
        pores)    envs="";                                         tag=_l$2 ;;
        poresk4)  envs="HOLES=1 HOLE_STIFFNESS=1e-4";              tag=_holes0.0001_l$2 ;;
        incl10)   envs="INCLUSION_STIFFNESS=10";                   tag=_incl10_l$2 ;;
        incl1000) envs="INCLUSION_STIFFNESS=1000";                 tag=_incl1000_l$2 ;;
    esac
    systemd-run --user --scope -p MemoryMax=4G -q env LENGTH_SCALE=$2 $envs \
        $PY -u example_2D_phase_field_fracture_AT2.py $3 > logs/AT2_N256_$3_polydisperse_pores${tag}_smooth.log 2>&1
    echo "$(date +%H:%M:%S) done $1 l=$2 $3"
}
export -f job; export PY
for P in Green_Jacobi Green dual; do for S in pores poresk4 incl1000 poresk3 incl10; do for L in 0.0078125 0.0625; do
    echo "$S $L $P"; done; done; done | xargs -P 12 -L 1 bash -c 'job $0 $1 $2'

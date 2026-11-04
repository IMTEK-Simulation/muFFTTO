#!/bin/bash
# dual at 512^2 on the GPU, one run after another: check of the extrapolation of the contrast study for the five
# material settings, sharp (l/h = 2, l = 1/256) and smooth (l = 1/16) line; output suffix _smooth. Skipped (they exist
# from run_overnight_2D_512_cpu.sh, suffix _lh): pores 1e-3 + damage and inclusions 10x at l/h = 2
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export MICROSTRUCTURE=polydisperse_pores MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu RUN_SUFFIX=_smooth NPIX=512 \
  DEVICE=gpu CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false
run() {  # SETTING L
    case $1 in
        poresk3)  envs="HOLES=1 HOLE_STIFFNESS=1e-3 K_RESIDUAL=1e-3"; tag=_holes0.001_l$2_k0.001 ;;
        pores)    envs="";                                         tag=_l$2 ;;
        poresk4)  envs="HOLES=1 HOLE_STIFFNESS=1e-4";              tag=_holes0.0001_l$2 ;;
        incl10)   envs="INCLUSION_STIFFNESS=10";                   tag=_incl10_l$2 ;;
        incl1000) envs="INCLUSION_STIFFNESS=1000";                 tag=_incl1000_l$2 ;;
    esac
    echo "$(date +%H:%M:%S) start $1 l=$2"
    systemd-run --user --scope -p MemoryMax=16G -q env LENGTH_SCALE=$2 $envs \
        $PY -u example_2D_phase_field_fracture_AT2.py dual > logs/AT2_N512_dual_polydisperse_pores${tag}_smooth.log 2>&1
    echo "$(date +%H:%M:%S) done  $1 l=$2 (exit $?)"
}
for S in poresk4 incl1000 poresk3; do run $S 0.00390625; done
for S in poresk4 incl1000 poresk3 pores incl10; do run $S 0.0625; done

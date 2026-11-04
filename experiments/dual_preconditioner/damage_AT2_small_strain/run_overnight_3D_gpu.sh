#!/bin/bash
# overnight 2026-10-31, GPU, one run after another: 3D AT2 at practical l/h = 2 (polydisperse balls, trilinear hexahedra)
#   1. stiff tough inclusions (10x), 32^3 (l = 1/16) and 64^3 (l = 1/32), Green / Green-Jacobi / dual
#   2. pores, 64^3 (l = 1/32); the 32^3 l/h = 2 point is the existing AT2_N32_*_polydisperse_pores_l0.0625_3D
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export MICROSTRUCTURE=polydisperse_pores DIM=3 DEVICE=gpu MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false
run() {  # N L INCL P
    tag=$([ -n "$3" ] && echo "_incl$3")_l$2
    echo "$(date +%H:%M:%S) start N=$1 l=$2 incl=$3 $4"
    systemd-run --user --scope -p MemoryMax=20G -q env NPIX=$1 LENGTH_SCALE=$2 ${3:+INCLUSION_STIFFNESS=$3} \
        $PY -u example_2D_phase_field_fracture_AT2.py $4 > logs/AT2_N$1_$4_polydisperse_pores${tag}_3D.log 2>&1
    echo "$(date +%H:%M:%S) done  N=$1 l=$2 incl=$3 $4 (exit $?)"
}
for P in dual Green_Jacobi Green; do run 32 0.0625 10 $P; done
for P in dual Green_Jacobi Green; do run 64 0.03125 10 $P; done
for P in Green_Jacobi dual Green; do run 64 0.03125 "" $P; done

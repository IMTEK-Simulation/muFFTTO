#!/bin/bash
# Gypsum RVE of run_AT2_gypsum_rve.sh (same parameters) on 512^2 (l/h = 10) and 1024^2 (l/h = 20), GPU, one run after
# another: 512^2 all preconditioners (grains first: short), then 1024^2 dual and Green-Jacobi and Green. Then the
# Cahn-Hilliard 512^2 runs of ../schneider2025_notch/run_AT2_cahn_hilliard_large_gpu.sh. Waits for the running GPU job.
# See README.md
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
while pgrep -f "bin/python -u example_2D_phase_field" > /dev/null; do sleep 30; done   # (pattern not in the launching shell)
GPU="DEVICE=gpu CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
gyp() {  # N SETTING P
    case $2 in
        pores) envs="HOLES=1 HOLE_STIFFNESS=1e-4"; tag=_holes0.0001 ;;
        grains) envs="INCLUSION_STIFFNESS=10";     tag=_incl10 ;;
    esac
    echo "$(date +%H:%M:%S) start gypsum N=$1 $2 $3"
    systemd-run --user --scope -p MemoryMax=20G -q env $GPU NPIX=$1 $envs MICROSTRUCTURE=polydisperse_pores \
        LENGTH_SCALE=0.0195915 GC=8.4635e-6 LAME=0.277778,0.416667 LOAD_MAX=0.015 LOAD_STEPS=150 RUN_SUFFIX=_gypsum \
        $PY -u example_2D_phase_field_fracture_AT2.py $3 > logs/AT2_N$1_$3_polydisperse_pores${tag}_l0.0195915_gypsum.log 2>&1
    echo "$(date +%H:%M:%S) done  gypsum N=$1 $2 $3 (exit $?)"
}
for P in dual Green_Jacobi Green; do gyp 512 grains $P; done
for P in dual Green Green_Jacobi; do gyp 512 pores $P; done
for S in grains pores; do for P in dual Green_Jacobi Green; do gyp 1024 $S $P; done; done
cd ../schneider2025_notch
for P in dual Green_Jacobi; do
    echo "$(date +%H:%M:%S) start CH N=512 $P"
    systemd-run --user --scope -p MemoryMax=20G -q env $GPU NPIX=512 GEOMETRY_FILE=geometry/cahn_hilliard_512.npy \
        INCLUSION_STIFFNESS=0.1 LENGTH_SCALE=0.03125 GC=1.2857e-8 LAME=0.576923,0.384615 LOAD_DIR=yy LOAD_MAX=2e-3 \
        LOAD_STEPS=200 RUN_SUFFIX=_ch K_RESIDUAL=1e-4 \
        $PY -u example_2D_phase_field_fracture_AT2.py $P > logs/AT2_N512_${P}_incl0.1_l0.03125_k0.0001_ch.log 2>&1
    echo "$(date +%H:%M:%S) done  CH N=512 $P (exit $?)"
done

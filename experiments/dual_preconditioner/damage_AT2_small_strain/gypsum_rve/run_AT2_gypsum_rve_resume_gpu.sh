#!/bin/bash
# Resume of run_AT2_gypsum_rve_1024_green_dual.sh after the reboot of 2026-11-03 18:46 (killed 1024^2 grains Green at
# load step 54; the Cahn-Hilliard runs never started). Order: Cahn-Hilliard 512^2 dual and Green-Jacobi of
# ../schneider2025_notch (short), then gypsum 1024^2 grains Green (long, > 1 day expected). Same parameters as before.
# Launch detached so that it survives the terminal / Claude session:
#   systemd-run --user --unit=gypsum-resume -p WorkingDirectory=$PWD systemd-inhibit --what=sleep:idle:handle-lid-switch \
#       --mode=block bash -c './run_AT2_gypsum_rve_resume_gpu.sh > logs/run_AT2_gypsum_rve_resume_gpu.txt 2>&1'
# See README.md
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
GPU="DEVICE=gpu CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
cd ../schneider2025_notch
for P in dual Green_Jacobi; do
    echo "$(date +%H:%M:%S) start CH N=512 $P"
    systemd-run --user --scope -p MemoryMax=20G -q env $GPU NPIX=512 GEOMETRY_FILE=geometry/cahn_hilliard_512.npy \
        INCLUSION_STIFFNESS=0.1 LENGTH_SCALE=0.03125 GC=1.2857e-8 LAME=0.576923,0.384615 LOAD_DIR=yy LOAD_MAX=2e-3 \
        LOAD_STEPS=200 RUN_SUFFIX=_ch K_RESIDUAL=1e-4 \
        $PY -u example_2D_phase_field_fracture_AT2.py $P > logs/AT2_N512_${P}_incl0.1_l0.03125_k0.0001_ch.log 2>&1
    echo "$(date +%H:%M:%S) done  CH N=512 $P (exit $?)"
done
cd ../gypsum_rve
echo "$(date +%H:%M:%S) start gypsum N=1024 grains Green"
systemd-run --user --scope -p MemoryMax=20G -q env $GPU NPIX=1024 INCLUSION_STIFFNESS=10 MICROSTRUCTURE=polydisperse_pores \
    LENGTH_SCALE=0.0195915 GC=8.4635e-6 LAME=0.277778,0.416667 LOAD_MAX=0.015 LOAD_STEPS=150 RUN_SUFFIX=_gypsum \
    $PY -u example_2D_phase_field_fracture_AT2.py Green > logs/AT2_N1024_Green_polydisperse_pores_incl10_l0.0195915_gypsum.log 2>&1
echo "$(date +%H:%M:%S) done  gypsum N=1024 grains Green (exit $?)"

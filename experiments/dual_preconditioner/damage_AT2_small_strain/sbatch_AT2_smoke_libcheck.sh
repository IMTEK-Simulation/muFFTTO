#!/bin/bash
# AT2 smoke test of a library change: the same short AT2 runs with the installed muFFTTO and with an older copy
# (PYTHONPATH=$OLDLIB, e.g. git archive <commit> muFFTTO), Green / Green-Jacobi / dual; the logs must agree
# (load-step table, CG counts) up to round-off.
#   sbatch [-p rtx] sbatch_AT2_smoke_libcheck.sh [OLDLIB=~/software/muFFTTO_oldlib_cae7082]
#   cases: 2D Cahn-Hilliard (ch settings of sbatch_AT2_gpu.sh) 128^2, 30 load steps;
#          3D stiff grains (polydisperse balls, INCLUSION_STIFFNESS 10) 32^3, l = 1/16, 20 load steps
# Logs: $MUFFTTO_WS/runs/AT2/smoke_libcheck/<case>_<precond>_{new,old}.log
#SBATCH --job-name=AT2_smoke_libcheck
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --output=/work/classic/fr_ml1145-muffto/runs/AT2/smoke_libcheck/slurm-%j.out
OLDLIB=${1:-$HOME/software/muFFTTO_oldlib_cae7082}
source ~/software/muffto_env_gpu.sh
export DEVICE=gpu CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-$CUDA_ROOT}} MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
OUT=$MUFFTTO_WS/runs/AT2/smoke_libcheck
mkdir -p "$OUT"
cd ~/software/muFFTTO/experiments/dual_preconditioner/damage_AT2_small_strain
run() {   # case lib precond
    local pp=""; [ "$2" = old ] && pp=$OLDLIB
    local log=$OUT/$1_$3_$2.log t0=$(date +%s)
    if [ "$1" = ch2D ]; then
        ( cd schneider2025_notch && PYTHONPATH=$pp GEOMETRY_FILE=geometry/cahn_hilliard_512.npy INCLUSION_STIFFNESS=0.1 \
          LENGTH_SCALE=0.03125 GC=1.2857e-8 LAME=0.576923,0.384615 LOAD_DIR=yy LOAD_MAX=2e-3 LOAD_STEPS=30 \
          K_RESIDUAL=1e-4 NPIX=128 RUN_SUFFIX=_smoke_$2 \
          python -u -c "import muFFTTO, runpy, sys; print('muFFTTO from', muFFTTO.__file__); sys.argv = ['x', '$3']; runpy.run_path('example_2D_phase_field_fracture_AT2.py', run_name='__main__')" ) > "$log" 2>&1
    else
        ( cd gypsum_rve && PYTHONPATH=$pp DIM=3 MICROSTRUCTURE=polydisperse_pores INCLUSION_STIFFNESS=10 \
          LENGTH_SCALE=0.0625 LOAD_STEPS=20 NPIX=32 RUN_SUFFIX=_smoke_$2 \
          python -u -c "import muFFTTO, runpy, sys; print('muFFTTO from', muFFTTO.__file__); sys.argv = ['x', '$3']; runpy.run_path('example_2D_phase_field_fracture_AT2.py', run_name='__main__')" ) > "$log" 2>&1
    fi
    echo "WALL $(( $(date +%s) - t0 )) s exit $?" >> "$log"
}
for c in ch2D grains3D; do for P in Green Green_Jacobi dual; do for lib in new old; do run $c $lib $P; done; done; done
cd "$OUT"
for c in ch2D grains3D; do for P in Green Green_Jacobi dual; do
    n=$(grep -v -e "muFFTTO from" -e WALL -e "elapsed\|time\|Time\|[0-9]\+\.[0-9]* s\b" ${c}_${P}_new.log | md5sum | cut -c1-8)
    o=$(grep -v -e "muFFTTO from" -e WALL -e "elapsed\|time\|Time\|[0-9]\+\.[0-9]* s\b" ${c}_${P}_old.log | md5sum | cut -c1-8)
    echo "$c $P new=$n old=$o $([ "$n" = "$o" ] && echo IDENTICAL || echo DIFFERENT)"
done; done | tee summary.txt

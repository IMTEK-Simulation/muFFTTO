#!/bin/bash
# Laptop prototype (CPU, one thread per run, all runs concurrent -> wall times comparable only within this batch):
# AT2 polydisperse pores, l/h = 2, MG vs dual.
#   64^2, 128^2: MG (library setup, 15 warm power iterations) and dual, d_min = 10 (standard energy test)
#   64^2 only:   MG with 3 warm power iterations (cheaper setup); MG and dual with d_min = 0 (no delay floor)
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu MPLBACKEND=Agg OMP_NUM_THREADS=1 PYTHONWARNINGS="ignore::RuntimeWarning"
PY=~/micromamba/envs/muffto-gpu/bin/python
run() {   # run <N> <precond> <suffix> [ENV=VALUE ...]
    local N=$1 P=$2 SUF=$3; shift 3
    local L=$($PY -c "print(2.0 / $N)")
    env MICROSTRUCTURE=polydisperse_pores NPIX=$N LENGTH_SCALE=$L RUN_SUFFIX=$SUF "$@" \
        $PY -u example_2D_phase_field_fracture_AT2.py $P > logs/AT2_N${N}_${P}_polydisperse_pores_lh2${SUF}.log 2>&1
}
for N in 64 128; do
    for P in MG dual; do run $N $P _lh2 & done
done
run 64 MG _lh2_pow3 MG_POWER=3 &
run 64 MG _lh2_d0 CG_DELAY=0 &
run 64 dual _lh2_d0 CG_DELAY=0 &
wait

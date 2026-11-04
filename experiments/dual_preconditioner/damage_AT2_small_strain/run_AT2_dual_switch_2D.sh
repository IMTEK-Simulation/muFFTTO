#!/bin/bash
# dual_switch (dual, in break steps the cheaper of dual (2 per iteration) / Green-Jacobi in CG iterations) on the 2D fixed-l/h problems at 128^2,
# against the dual / Green-Jacobi runs of run_AT2_incl_lh_128.sh (suffix _lh)
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export MICROSTRUCTURE=polydisperse_pores MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu RUN_SUFFIX=_lh NPIX=128
run() {  # L INCL
    tag=$([ -n "$2" ] && echo "_incl$2")_l$1
    systemd-run --user --scope -p MemoryMax=8G -q env LENGTH_SCALE=$1 ${2:+INCLUSION_STIFFNESS=$2} \
        $PY -u example_2D_phase_field_fracture_AT2.py dual_switch > logs/AT2_N128_dual_switch_polydisperse_pores${tag}_lh.log 2>&1
}
run 0.015625 10 & run 0.015625 "" & run 0.03125 10 & run 0.03125 "" &
wait

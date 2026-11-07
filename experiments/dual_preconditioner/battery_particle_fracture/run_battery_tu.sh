#!/bin/bash
# Tu et al. 2026 length scales: l = 0.23 / 0.27 um (core / shell), l_zeta = 0.1 um, h/R = 0.2, a/h = 0.3;
# particle R_core = 2 um (a value of the paper's R study) in a 6 um periodic cell -> 64^2: h = 0.094 um (l/h >= 2.45),
# 128^2: h = 0.047 um. 32^2 cannot resolve l (l/h = 1.2): not run. dt = 22 s, until the CV cut-off (T_MAX = 4969 s).
#   ./run_battery_tu.sh Green & ./run_battery_tu.sh Green_Jacobi & ./run_battery_tu.sh dual &
cd "$(dirname "$0")"
P=$1
SHADOW=0; [ "$P" = dual ] && SHADOW=1
for N in ${GRIDS:-64 128}; do
  systemd-run --user --scope -p MemoryMax=4G -q env MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
    NPIX=$N L_CELL_UM=6 R_CORE_UM=2 H_BAR=0.2 A_BAR=0.3 L_CORE_UM=0.23 L_SHELL_UM=0.27 L_ZETA_UM=0.1 \
    DT=0.02 T_MAX=4.5 MATRIX_STIFFNESS=${MATRIX_STIFFNESS:-1e-2} DIFF_SHADOW=$SHADOW SNAP_EVERY=${SNAP_EVERY:-10} RUN_SUFFIX=_tu \
    ~/micromamba/envs/muffto-gpu/bin/python -u example_2D_battery_particle_fracture_AT2.py $P \
    > logs/battery_tu_N${N}_${P}.log 2>&1
done

#!/bin/bash
# All 2D experiments of this folder with the plain trust-region Newton-CG (every increment starts from the last
# converged state). Logs into exp_data/<script>/log_<name>.txt
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu
PY=~/micromamba/envs/muffto-gpu/bin/python
run() {  # SCRIPT LOG_NAME, then environment assignments
    local script=$1 log=$2; shift 2
    mkdir -p exp_data/$script
    env "$@" $PY -u $script.py 2>&1 | grep -vE "Warning|laplacian|dirac_comb" > exp_data/$script/log_$log.txt
    echo "$script $log: $(grep -E 'total TR|step too small|Traceback' exp_data/$script/log_$log.txt | tail -1)"
}
# point-load box (Figs. 3, 4, Tables 1, 2)
run box_point_load n40_gamma2e-4_v1
run box_point_load n40_gamma2e-4_v1_decoupled MATERIAL=decoupled
for gamma in 1e-4 1e-5 1e-6 1e-7; do run box_point_load n40_gamma${gamma}_v0.75 GAMMA=$gamma V_MAX=0.75 NB_INCREMENTS=15; done
for n in 20 80; do run box_point_load n${n}_gamma1e-6_v0.75 NB_PER_UNIT=$n GAMMA=1e-6 V_MAX=0.75 NB_INCREMENTS=15; done
# C-box (Figs. 11-13)
run cbox_point_load n40_gamma1e-6
# pressure box (Figs. 6, 7)
run box_pressure path_gamma1e-4_sym GAMMA=1e-4 SYMMETRIC=1
run box_pressure path_gamma1e-4 GAMMA=1e-4 SYMMETRIC=0
run box_pressure paper_P0.1_gamma1e-5 GAMMA=1e-5 PRESSURE_PATH=0,0.1 PRESSURE_STEP=0.005 SYMMETRIC=1
run box_pressure paper_P1_gamma1e-4_sym GAMMA=1e-4 PRESSURE_PATH=0,1 SYMMETRIC=1
run box_pressure paper_P-0.8_gamma5e-4 GAMMA=5e-4 PRESSURE_PATH=0,-0.8 SYMMETRIC=1

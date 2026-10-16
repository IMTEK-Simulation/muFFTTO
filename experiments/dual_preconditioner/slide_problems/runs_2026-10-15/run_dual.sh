#!/bin/bash
# dual preconditioner on the four slide problems, 16^2 / 32^2 / 64^2, adapted contact + fracture first;
# 7 lanes of run_set.sh (2 threads each), progress in run_dual.log
# Record of 2026-10-15: run from the slides folder before the move (the run.py files write to
# examples/phase_field_fracture_adapted_grid/exp_data/slides/runs).
cd "$(dirname "$0")"
./run_set.sh N16_adapted_dual 3_contact 4_fracture && ./run_set.sh N16_dual 2_trust_region 3_contact 4_fracture &
./run_set.sh N32_adapted_dual 3_contact &
./run_set.sh N32_adapted_dual 4_fracture &
./run_set.sh N64_adapted_dual 3_contact &
./run_set.sh N64_adapted_dual 4_fracture &
./run_set.sh N32_dual 3_contact 4_fracture &
./run_set.sh N64_dual 2_trust_region 3_contact 4_fracture &
wait

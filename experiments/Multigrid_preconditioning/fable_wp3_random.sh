#!/usr/bin/env bash
# WP3: symmetric 32^3 TO from random cubic-symmetric starts -> logs/fable/wp3_random_report.txt
cd "$(dirname "$0")"; source ./fable_env.sh
$PY fable_make_random_sym_starts.py > logs/fable/wp3_random_made.txt 2>&1
export SCRIPT=to3d_large.py DESIGN_OPT=lbfgs_helm_local SYMMETRY=cubic
for K in 2 3; do for S in 0 1 2 3; do
  INIT_FROM=$PWD/data/CH3D/sym_start_rnd${K}_s$S.nc ./fable_run_TO.sh symrnd${K}_s$S 32 400
done; done
cd ../dual_preconditioner/topology_optimization
$PY report_TO3D_nu.py ../../Multigrid_preconditioning/data/to3d_large/N32_*_fable_symrnd*_final.nc \
    > ../../Multigrid_preconditioning/logs/fable/wp3_random_report.txt 2>&1

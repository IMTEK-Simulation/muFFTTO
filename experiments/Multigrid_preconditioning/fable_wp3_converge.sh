#!/usr/bin/env bash
# WP3: 32^3 TO to convergence with SYMMETRY=cubic and without (same start, 6 load cases); then Poisson ratios
cd "$(dirname "$0")"; source ./fable_env.sh
export SCRIPT=to3d_large.py DESIGN_OPT=lbfgs_helm_local
SYMMETRY=cubic ./fable_run_TO.sh conv_sym 32 400
./fable_run_TO.sh conv_full 32 400
cd ../dual_preconditioner/topology_optimization
$PY report_TO3D_nu.py ../../Multigrid_preconditioning/data/to3d_large/N32_*_fable_conv_sym_final.nc \
    ../../Multigrid_preconditioning/data/to3d_large/N32_*_fable_conv_full_final.nc \
    > ../../Multigrid_preconditioning/logs/fable/wp3_nu_report.txt 2>&1

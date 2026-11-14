#!/usr/bin/env bash
# WP3: symmetric 32^3 TO from several cubic-symmetric starts -> logs/fable/wp3_starts_report.txt
cd "$(dirname "$0")"; source ./fable_env.sh
$PY fable_make_sym_starts.py data/to3d_large/N32_soft0.001_eta0.0625_energy_fable_conv_full_final.nc > logs/fable/wp3_starts_made.txt 2>&1
export SCRIPT=to3d_large.py DESIGN_OPT=lbfgs_helm_local SYMMETRY=cubic
for S in cos2 cos1 prod1 edges fromfull fromfull_sharp; do
  INIT_FROM=$PWD/data/CH3D/sym_start_$S.nc ./fable_run_TO.sh symstart_$S 32 400
done
cd ../dual_preconditioner/topology_optimization
$PY report_TO3D_nu.py ../../Multigrid_preconditioning/data/to3d_large/N32_*_fable_symstart_*_final.nc \
    > ../../Multigrid_preconditioning/logs/fable/wp3_starts_report.txt 2>&1

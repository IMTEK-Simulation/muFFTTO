#!/usr/bin/env bash
# WP3: 32^3 TO with SYMMETRY=cubic_chiral (24 rotations, no mirrors) from several starts -> logs/fable/wp3_chiral_report.txt
cd "$(dirname "$0")"; source ./fable_env.sh
$PY fable_make_random_sym_starts.py > logs/fable/wp3_random_made.txt 2>&1
export SCRIPT=to3d_large.py DESIGN_OPT=lbfgs_helm_local SYMMETRY=cubic_chiral
./fable_run_TO.sh chiral_CH 32 400
INIT_FROM=$PWD/data/to3d_large/N32_soft0.001_eta0.0625_energy_fable_conv_full_final.nc ./fable_run_TO.sh chiral_fromfull 32 400
for K in 2 3; do for S in 0 1; do
  INIT_FROM=$PWD/data/CH3D/chiral_start_rnd${K}_s$S.nc ./fable_run_TO.sh chiral_rnd${K}_s$S 32 400
done; done
cd ../dual_preconditioner/topology_optimization
$PY report_TO3D_nu.py ../../Multigrid_preconditioning/data/to3d_large/N32_*_fable_chiral_*_final.nc \
    > ../../Multigrid_preconditioning/logs/fable/wp3_chiral_report.txt 2>&1

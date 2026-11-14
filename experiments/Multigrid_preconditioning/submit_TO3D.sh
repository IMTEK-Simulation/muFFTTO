#!/bin/bash
# 3D TO with MG (overnight 2026-11-13): eta = 4h hierarchy 64^3 (l40s) -> 128^3 (h200), new 3D CH start bi_s11.
# MG + Helmholtz c = 4 vs production Green-Jacobi + NuMPI L-BFGS; MG + NuMPI L-BFGS at 64^3. Smoke job first.
cd ~/software/muFFTTO/experiments/Multigrid_preconditioning
R=/work/classic/fr_ml1145-muffto/runs; JL=$R/MG_overnight_jobs.txt
D=$PWD/data/example_3D_elasticity_TO
START=$PWD/data/CH3D/CH3D_start_bi_s11.nc
C="SCRIPT=example_3D_elasticity_TO.py,ETA_H=4,LABEL=bi3d_s11"
S=$(sbatch --parsable --time=00:30:00 --export=ALL,$C,DESIGN_OPT=lbfgs_helm,HELM_C=4,MAXITER=2,LOG_SUFFIX=_smoke3d sbatch_TO_mg.sh MG 16 $START)
echo "$S smoke TO3D MG 16" | tee -a $JL
A=$(sbatch --parsable --time=12:00:00 --dependency=afterok:$S --export=ALL,$C,DESIGN_OPT=lbfgs_helm,HELM_C=4 sbatch_TO_mg.sh MG 64 $START)
B=$(sbatch --parsable -p h200 --mem=96G --time=24:00:00 --dependency=afterok:$A --export=ALL,$C,DESIGN_OPT=lbfgs_helm,HELM_C=4 sbatch_TO_mg.sh MG 128 $D/N64_soft0.0001_energy-warm_MG_eta_0.0625_pfw0.2_l_bfgs_mgTO_bi3d_s11_lbfgs_helm_c4_final.nc)
echo "$A $B TO3D chain MG + Helmholtz c4 64 -> 128(h200)" | tee -a $JL
G1=$(sbatch --parsable --time=24:00:00 --dependency=afterok:$S --export=ALL,$C sbatch_TO_mg.sh Green_Jacobi 64 $START)
G2=$(sbatch --parsable -p h200 --mem=96G --time=24:00:00 --dependency=afterok:$G1 --export=ALL,$C sbatch_TO_mg.sh Green_Jacobi 128 $D/N64_soft0.0001_energy-warm_Green_Jacobi_eta_0.0625_pfw0.2_l_bfgs_mgTO_bi3d_s11_final.nc)
echo "$G1 $G2 TO3D chain GJ + NuMPI L-BFGS 64 -> 128(h200)" | tee -a $JL
M=$(sbatch --parsable --time=12:00:00 --dependency=afterok:$S --export=ALL,$C sbatch_TO_mg.sh MG 64 $START)
echo "$M TO3D MG + NuMPI L-BFGS 64" | tee -a $JL

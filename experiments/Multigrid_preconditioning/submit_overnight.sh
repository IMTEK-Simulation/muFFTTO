#!/bin/bash
# Overnight batch 2026-11-13 (user asleep; asked for: 3D MG scaling 32/64/128 first, new random Cahn-Hilliard
# structures, large 2D runs 2048^2 "and so on" on the H200s). Smoke jobs first; everything depends on them (afterok).
#   bash submit_overnight.sh        (on NEMO, login node: only sbatch calls)
cd ~/software/muFFTTO/experiments/Multigrid_preconditioning
R=/work/classic/fr_ml1145-muffto/runs
JL=$R/MG_overnight_jobs.txt
D=$PWD/data/example_2D_elasticity_TO
echo "# submitted $(date -Is)" >> $JL
# ---- smoke: 3D driver (l40s), TO + mg_lib in the h200 environment
S3=$(sbatch --parsable --time=00:30:00 --export=ALL,OUT=$R/MG_scaling3d_smoketest sbatch_mg_scaling_3d.sh ch_s12 16 "0 1")
SH=$(sbatch --parsable -p h200 --time=00:20:00 --export=ALL,DESIGN_OPT=lbfgs_helm,HELM_C=4,MAXITER=3,LOG_SUFFIX=_smoke_h200 sbatch_TO_mg.sh MG 64)
echo "$S3 smoke 3D l40s | $SH smoke TO MG h200" | tee -a $JL
# ---- 3D scaling (resolved eta only: 32: eta/h <= 1, 64: <= 2, 128: <= 4)
for spec in "ch_s12 32 0 0.5 1" "ch_s12 64 0 0.5 1 2" "bi_s11 32 0 1" "bi_s11 64 0 1" "ch_s13 32 0 1" "ch_s13 64 0 1"; do
  set -- $spec; G=$1; N=$2; shift 2
  J=$(sbatch --parsable --dependency=afterok:$S3 sbatch_mg_scaling_3d.sh $G $N "$*"); echo "$J 3D $G $N eta $* l40s" | tee -a $JL
done
for spec in "ch_s12 128 0 0.5 1 2 4" "bi_s11 128 0 1" "ch_s13 128 0 1"; do
  set -- $spec; G=$1; N=$2; shift 2
  J=$(sbatch --parsable -p h200 --mem=96G --dependency=afterok:$S3:$SH sbatch_mg_scaling_3d.sh $G $N "$*"); echo "$J 3D $G $N eta $* h200" | tee -a $JL
done
# ---- 2D TO chains: eta = 4h hierarchy, MG + Helmholtz L-BFGS (c = 4), new CH starts; INIT_FROM = previous final design
fin() { python3 -c "print('$D/N$1_soft0.0001_energy-warm_MG_eta_' + str(4 / $1) + '_pfw0.2_l_bfgs_mgTO_$2_lbfgs_helm_c4_final.nc')"; }
for L in bi_s11 bi_s12 drop_s11; do
  START=$PWD/data/CH_starts_new/CH_start_$L.nc
  E="ETA_H=4,DESIGN_OPT=lbfgs_helm,HELM_C=4,LABEL=$L"
  J1=$(sbatch --parsable --time=06:00:00 --export=ALL,$E sbatch_TO_mg.sh MG 256 $START)
  J2=$(sbatch --parsable --time=12:00:00 --dependency=afterok:$J1 --export=ALL,$E sbatch_TO_mg.sh MG 512 $(fin 256 $L))
  J3=$(sbatch --parsable -p h200 --mem=64G --time=12:00:00 --dependency=afterok:$J2:$SH --export=ALL,$E sbatch_TO_mg.sh MG 1024 $(fin 512 $L))
  J4=$(sbatch --parsable -p h200 --mem=96G --time=24:00:00 --dependency=afterok:$J3 --export=ALL,$E sbatch_TO_mg.sh MG 2048 $(fin 1024 $L))
  echo "$J1 $J2 $J3 $J4 TO chain $L 256 512 1024(h200) 2048(h200)" | tee -a $JL
  if [ $L = bi_s11 ]; then
    J5=$(sbatch --parsable -p h200 --mem=128G --time=24:00:00 --dependency=afterok:$J4 --export=ALL,$E sbatch_TO_mg.sh MG 4096 $(fin 2048 $L))
    echo "$J5 TO chain $L 4096(h200)" | tee -a $JL
  fi
  B=$(sbatch --parsable --time=06:00:00 --export=ALL,ETA_H=4,LABEL=$L sbatch_TO_mg.sh Green_Jacobi 256 $START)
  echo "$B TO baseline GJ + NuMPI L-BFGS $L 256" | tee -a $JL
done

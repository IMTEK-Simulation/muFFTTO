#!/bin/bash
# NEMO results -> laptop: iterations CSVs, logs, the small final figures and the final damage of the dual (d_min 10)
# sweep runs (never the .nc files: 1024^2 ~4.5 GB each)
cd "$(dirname "$0")"
R=nemo:/work/classic/fr_ml1145-muffto/runs/MG_vs_dual
mkdir -p exp_data/nemo logs/nemo figures/nemo
rsync -a "$R/exp_data/" exp_data/nemo/ --include '*_iterations.csv' --exclude '*_d0_final_damage.npy' \
      --include '*_dual_polydisperse_pores_*_sw_lh*_final_damage.npy' --exclude '*'
rsync -a "$R/logs/" logs/nemo/ --include '*.log' --exclude '*'
rsync -a "$R/figures/" figures/nemo/ --include '*.png' --exclude '*'
rsync -a "$R/jobs.txt" logs/nemo/jobs.txt

#!/bin/bash
# fetch the NEMO scaling CSVs, logs and job list -> exp_data/nemo_scaling/
cd "$(dirname "$0")"
R=/work/classic/fr_ml1145-muffto/runs/MG_scaling
mkdir -p exp_data/nemo_scaling
timeout 300 rsync -a nemo:$R/csv nemo:$R/logs nemo:$R/jobs.txt exp_data/nemo_scaling/

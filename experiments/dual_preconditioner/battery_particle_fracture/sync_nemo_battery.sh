#!/bin/bash
# Copy the NEMO battery CSVs and logs (no .npz/.nc) to exp_data/nemo, logs/nemo, and the queue state
cd "$(dirname "$0")"
mkdir -p exp_data/nemo logs/nemo
R=/work/classic/fr_ml1145-muffto/runs/battery
timeout 300 rsync -a --include='*.csv' --exclude='*' nemo:$R/exp_data/ exp_data/nemo/
timeout 300 rsync -a nemo:$R/logs/ logs/nemo/
timeout 60 rsync -a nemo:$R/jobs.txt logs/nemo/jobs.txt

#!/bin/bash
# fetch the NEMO TO logs and job list -> logs/nemo_TO/ (designs .nc stay on NEMO)
cd "$(dirname "$0")"
mkdir -p logs/nemo_TO
timeout 300 rsync -a --include='*.log' --include='jobs.txt' --exclude='*' nemo:/work/classic/fr_ml1145-muffto/runs/MG_TO/ logs/nemo_TO/

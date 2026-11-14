#!/bin/bash
# fetch the NEMO 3D scaling CSVs and logs -> exp_data/nemo_scaling3d/
cd "$(dirname "$0")"
R=/work/classic/fr_ml1145-muffto/runs/MG_scaling3d
mkdir -p exp_data/nemo_scaling3d
timeout 300 rsync -a nemo:$R/csv nemo:$R/logs exp_data/nemo_scaling3d/
timeout 60 rsync -a nemo:/work/classic/fr_ml1145-muffto/runs/MG_overnight_jobs.txt exp_data/nemo_scaling3d/ 2>/dev/null

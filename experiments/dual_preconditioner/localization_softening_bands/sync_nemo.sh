#!/bin/bash
# copy the PF-CZM mesh-study results back from NEMO: CSVs, phase masks, logs, 2D frames (every 10th step, dual only),
# 3D mid-plane frames; no checkpoints, no full 3D fields
cd "$(dirname "$0")"
R=nemo:/work/classic/fr_ml1145-muffto/runs/czm
mkdir -p exp_data_nemo logs_nemo
rsync -a -e 'ssh -o BatchMode=yes' --include='*/' --include='*.csv' --include='*_phase.npy' \
    --include='PFCZM2D_*_dual_*_frames/d_*0.npy' --include='PFCZM3D_*_frames/d_mid_*.npy' --exclude='*' \
    $R/exp_data/ exp_data_nemo/
rsync -a -e 'ssh -o BatchMode=yes' $R/logs/ $R/jobs.txt logs_nemo/

#!/bin/bash
# copy the NEMO ASR mesh-study results (CSV, steps npz, 2D / mid-plane damage snapshots, logs, job list) to the laptop;
# no checkpoints, no 3D full fields
cd "$(dirname "$0")"
R=/work/classic/fr_ml1145-muffto/runs/asr
rsync -a --exclude='*_checkpoint*' --exclude='*_full.npy' --include='ASRmesh_*' --include='ASRmesh_*_damage/***' \
      --exclude='*' nemo:$R/exp_data/ exp_data/
rsync -a nemo:$R/logs/ logs/nemo/
rsync -a nemo:$R/jobs.txt logs/nemo/jobs.txt

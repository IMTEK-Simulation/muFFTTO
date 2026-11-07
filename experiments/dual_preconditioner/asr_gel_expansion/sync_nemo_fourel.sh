#!/bin/bash
# fetch the NEMO Fourel-model results (CSV, steps npz, 2D / mid-plane damage snapshots, logs, slurm outputs, job list)
# into exp_data/ and logs/nemo_fourel/; no checkpoints, no 3D full fields (FULL=1 also fetches *_full.npy)
cd "$(dirname "$0")"
R=/work/classic/fr_ml1145-muffto/runs/asr_fourel
mkdir -p exp_data logs/nemo_fourel
ex=(--exclude='*_checkpoint*')
[ "${FULL:-0}" = 1 ] || ex+=(--exclude='*_full.npy')
rsync -a "${ex[@]}" --include='FOUREL*' --include='FOUREL*_damage/***' --exclude='*' nemo:$R/exp_data/ exp_data/
rsync -a nemo:$R/logs/ logs/nemo_fourel/
rsync -a nemo:$R/slurm/ logs/nemo_fourel/slurm/
rsync -a nemo:$R/jobs.txt logs/nemo_fourel/jobs.txt 2>/dev/null || true

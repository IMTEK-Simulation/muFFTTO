#!/bin/bash
# asr_pf_fourel.py on NEMO2 (over the user's ssh ControlMaster: ssh -o BatchMode=yes nemo)
#   ./sync_nemo_asrpf.sh push    scripts + geometry -> ~/software/muFFTTO/experiments/dual_preconditioner/asr_gel_expansion
#   ./sync_nemo_asrpf.sh fetch   results (CSV, steps npz, 2D / mid-plane snapshots, logs, job list) ->
#                                exp_data/nemo_pf/, logs/nemo_pf/ (no checkpoints, no 3D full fields)
cd "$(dirname "$0")"
R=/work/classic/fr_ml1145-muffto/runs/asr_pf_fourel
E=software/muFFTTO/experiments/dual_preconditioner/asr_gel_expansion
SSH="ssh -o BatchMode=yes"
case $1 in
  push)
    $SSH nemo "mkdir -p $R/exp_data $R/logs $R/slurm ~/$E/geometry"
    rsync -a -e "$SSH" asr_pf_fourel.py make_fourel_geometry_2D.py make_fourel_geometry_3D.py sbatch_asrpf_gpu.sh \
          nemo:$E/
    rsync -a -e "$SSH" geometry/fourel2D_s1.npz $( [ -f geometry/fourel3D_s1.npz ] && echo geometry/fourel3D_s1.npz ) \
          nemo:$E/geometry/ ;;
  fetch)
    mkdir -p logs/nemo_pf exp_data/nemo_pf
    rsync -a -e "$SSH" --exclude='*_checkpoint*' --exclude='*_full.npy' --include='ASRpf_*' \
          --include='ASRpf_*_damage/***' --exclude='*' nemo:$R/exp_data/ exp_data/nemo_pf/
    rsync -a -e "$SSH" nemo:$R/logs/ logs/nemo_pf/
    rsync -a -e "$SSH" nemo:$R/slurm/ logs/nemo_pf/slurm/
    rsync -a -e "$SSH" nemo:$R/jobs.txt logs/nemo_pf/jobs.txt 2>/dev/null || true ;;
  *) echo "usage: $0 push|fetch"; exit 1 ;;
esac

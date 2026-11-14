#!/bin/bash
# Render one AT1 video on a NEMO2 CPU node (the 1024^2 .nc files stay on the workspace):
#   sbatch [--dependency=afterok:<job>] sbatch_AT1_video.sh <N> <P> <TAG> [FIXED_LH]
#   e.g. sbatch sbatch_AT1_video.sh 1024 dual _polydisperse_pores_l0.03125
#   STRESS_ONLY=1 GRID_STUDY=1 ... (environment, passed through) -> the slide layout with the grid study panel
# Output: figures/AT1_N<N>_<P><TAG>[...].mp4 + _final.png (figures -> $WS/runs/AT1/figures)
#SBATCH --job-name=AT1_video
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/work/classic/fr_ml1145-muffto/logs/AT1_video-%j.out
set -e
source ~/software/muffto_env.sh
module load vis/ffmpeg 2>/dev/null || module load vis/ffmpeg/6.0-gcccore-12.3.0
which ffmpeg
export MPLBACKEND=Agg OMP_NUM_THREADS=2
[ -n "$4" ] && export FIXED_LH=$4
cd ~/software/muFFTTO/experiments/dual_preconditioner/damage_AT1_small_strain
echo "job $SLURM_JOB_ID host $(hostname) start $(date -Is) N=$1 P=$2 TAG=$3 FIXED_LH=${FIXED_LH:-} STRESS_ONLY=${STRESS_ONLY:-} GRID_STUDY=${GRID_STUDY:-}"
python -u make_AT1_video.py "$1" "$2" "$3"
echo "done $(date -Is)"

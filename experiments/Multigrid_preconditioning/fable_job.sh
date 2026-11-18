#!/bin/bash
# Generic NEMO2 job of the large-3D-TO study: runs the given command line in this folder of the clone
# ~/software/muFFTTO_fable. Partition, GPUs, time and name are chosen on the sbatch command line:
#   sbatch -J FB_<name> [-p h200] [--gres=gpu:N --ntasks=N] [--time=..] [--mem=..] fable_job.sh <command> [args...]
#   -> $WS/runs/FABLE/logs/<name>-<jobid>.out
#SBATCH --job-name=FB_job
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --exclude=g3606,g3608         # GPU ECC errors
#SBATCH --output=/work/classic/fr_ml1145-muffto/runs/FABLE/logs/%x-%j.out
cd ~/software/muFFTTO_fable/experiments/Multigrid_preconditioning
echo "job $SLURM_JOB_ID $(hostname) $(date -Is) partition $SLURM_JOB_PARTITION: $*"; nvidia-smi -L
"$@"
echo "WALL done $(date -Is) exit $?"

#!/usr/bin/env bash
# ghost-exchange benchmark under several UCX transport settings (inside an sbatch job with >= 2 GPUs)
cd "$(dirname "$0")"; source ./fable_env.sh
N=${1:-256}; NP=${SLURM_NTASKS:-2}
run() { echo "--- $*"; env "$@" srun --ntasks=$NP --gpus-per-task=1 --cpus-per-task=${SLURM_CPUS_PER_TASK:-2} $PY fable_time_ghosts.py $N 2>&1 | grep -v -i warn; }
srun --ntasks=1 --gpus-per-task=1 $PY fable_time_ghosts.py $N 2>&1 | grep -v -i warn
run UCX_LOG_LEVEL=warn
ucx_info -v 2>&1 | head -3; ucx_info -d 2>/dev/null | grep -E "Transport: (cuda|sysv|posix|cma|knem|xpmem)" | sort | uniq -c

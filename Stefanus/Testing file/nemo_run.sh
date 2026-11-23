#!/bin/bash
# NEMO2: run the TR_SRI script. Everything after the script name goes to Python:
#   GPU test:    sbatch -p h200 --gres=gpu:1 -t 00:30:00 nemo_run.sh --solver cg --device gpu -n 32 -inc 2 --save_per_it
#   direct:      sbatch -p cpu nemo_run.sh --solver direct -n 64 -inc 40
#   MPI CG:      sbatch -p cpu --ntasks=8 nemo_run.sh --solver cg -n 128 -inc 40
#   other file:  SCRIPT=TR_SRI_debug_gpu.py sbatch -p h200 --gres=gpu:1 -t 00:10:00 nemo_run.sh --solver cg --device gpu -n 32 -inc 2
# The venv of the partition must exist (nemo_build_env.sh on that partition).
#SBATCH --job-name=tr_sri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=tr_sri_%j.out
set -eo pipefail

P=${SLURM_JOB_PARTITION:?run this as a Slurm job: sbatch -p <partition> nemo_run.sh ...}
SCRIPT=${SCRIPT:-TR_SRI_2D_homogenization_finite_strain_elasticity_NeoHookean.py}
SCRIPT_DIR="$HOME/thesis/muFFTTO/Stefanus/Testing file"
VENV=$HOME/venvs/muffto-$P
[ -d "$VENV" ] || { echo "no venv for partition $P: first run  sbatch -p $P nemo_build_env.sh"; exit 1; }

module purge
module load lang/python/3.13.5-gcccore-14.3.0 mpi/openmpi/5.0.8-gcc-14.3.0
case $P in
    l40s|h200|rtx)
        module load system/cuda/12.9.1
        export CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-${CUDA_ROOT:-}}}
        nvidia-smi -L ;;
esac
source "$VENV/bin/activate"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$SCRIPT_DIR"
echo "== partition $P | host $(hostname) | MPI tasks ${SLURM_NTASKS:-1} | python -u $SCRIPT $* | $(date -Is)"
if [ "${SLURM_NTASKS:-1}" -gt 1 ]; then
    srun python -u "$SCRIPT" "$@"
else
    python -u "$SCRIPT" "$@"
fi
echo "== done $(date -Is)"

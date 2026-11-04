#!/bin/bash
# MPI x GPU test of example_3D_elasticity_TO.py on one NEMO2 L40S node: the same short run (N^3, 3 L-BFGS iterations)
# on 1, 2 and 4 GPUs (one MPI rank per GPU, srun --gpus-per-task=1, so every rank sees its GPU as device 0).
# The objective, gradient norm and CG counts of the multi-GPU runs must equal the 1-GPU run up to round-off.
#   sbatch sbatch_TO3D_mpi_gpu_test.sh [N=64]
#   INIT_NOISE_GRID=<N>: start noise independent of the MPI split (needed to compare runs on different GPU counts)
#   MPI transport of device buffers: --export=ALL,MUGRID_GPU_AWARE_MPI=0 (host staging) or EXTRA_MODULES=lib/ucx-cuda/...
#   fewer GPUs / another partition: sbatch -p rtx --gres=gpu:2 --ntasks=2 --export=ALL,RUNS="Green_Jacobi 1;Green_Jacobi 2;dual 2",TAG=rtx ...
# Logs: $MUFFTTO_WS/runs/TO3D/mpi_test/<precond>_np<k>[_<TAG>].log
#SBATCH --job-name=TO3D_mpi_gpu_test
#SBATCH --partition=l40s
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=/work/classic/fr_ml1145-muffto/runs/TO3D/mpi_test/slurm-%j.out
N=${1:-64}
cd ~/software/muFFTTO/experiments/dual_preconditioner/topology_optimization
source ~/software/muffto_env_gpu.sh
[ -n "${EXTRA_MODULES:-}" ] && module load $EXTRA_MODULES     # e.g. lib/ucx-cuda/1.19.0-gcccore-14.3.0-cuda-12.9.1
export DEVICE=gpu CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-$CUDA_ROOT}} OMP_NUM_THREADS=1
OUT=$MUFFTTO_WS/runs/TO3D/mpi_test
mkdir -p "$OUT"
T=${TAG:+_$TAG}
python -c "import muGrid, mpi4py; print('muGrid', muGrid.__version__, 'has_mpi', getattr(muGrid, 'has_mpi', '?'),
      'gpu_aware_mpi', getattr(muGrid._muGrid.Communicator, 'gpu_aware_mpi', '?')); print(mpi4py.get_config())" \
    > "$OUT/env$T.txt" 2>&1
echo "MUGRID_GPU_AWARE_MPI=${MUGRID_GPU_AWARE_MPI:-unset} EXTRA_MODULES=${EXTRA_MODULES:-}" >> "$OUT/env$T.txt"
ompi_info --parsable --all 2>/dev/null | grep -i "mpi_built_with_cuda_support:value" >> "$OUT/env$T.txt"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader >> "$OUT/env$T.txt"
IFS=';' read -ra RUNLIST <<< "${RUNS:-Green_Jacobi 1;Green_Jacobi 2;Green_Jacobi 4;dual 1;dual 2}"
for run in "${RUNLIST[@]}"; do
    set -- $run; P=$1; NP=$2              # optional 3rd field: grid size of this run (default N)
    LOG=$OUT/${P}_np${NP}_N${3:-$N}$T.log
    start=$(date +%s)
    RUN_SUFFIX=_mpitest_np$NP$T MAXITER=${MAXITER:-3} NPIX=${3:-$N} INIT_NOISE_GRID=${INIT_NOISE_GRID:-0} SOFT_EXP=4 ETA=0.0625 CG_MODE=energy GTOL=0 HISTORY_EVERY=1 \
        timeout 1200 srun --exact -n "$NP" --gpus-per-task=1 --cpus-per-task=2 \
        python -u example_3D_elasticity_TO.py "$P" > "$LOG" 2>&1
    echo "WALL $(( $(date +%s) - start )) s exit $?" >> "$LOG"
done

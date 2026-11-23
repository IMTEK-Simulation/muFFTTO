#!/bin/bash
# NEMO2: one-time build of the Python environment for the TR_SRI script, for ONE partition, on a node of that partition.
# Every NEMO2 partition has its own software stack compiled for its CPU: a venv built on one partition can die with
# 'Illegal instruction' on another (Martin, nemo_build_gpu_env.sh). So build once per partition you want to use:
#     GPU:  sbatch -p h200 --gres=gpu:1 nemo_build_env.sh      (or -p l40s)
#     CPU:  sbatch -p cpu  nemo_build_env.sh                   (direct solver, MPI CG)
# Result: $HOME/venvs/muffto-<partition> with numpy, scipy, matplotlib, jax (CPU), mpi4py (cluster OpenMPI),
#         muGrid 1.6.1 (MPI, + CUDA on GPU partitions), CuPy (GPU partitions), NuMPI feat/tr_precond, muFFTTO (editable)
#SBATCH --job-name=build_muffto
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=build_muffto_%j.out
set -eo pipefail   # no -u: the Lmod module scripts use unset variables

P=${SLURM_JOB_PARTITION:?run this as a Slurm job: sbatch -p <partition> nemo_build_env.sh}
case $P in
    l40s) CUDA_ARCH=89 ;;
    h200) CUDA_ARCH=90 ;;
    rtx)  CUDA_ARCH=120 ;;
    mi300a) echo "mi300a is an AMD GPU (ROCm): not supported by this script"; exit 1 ;;
    *)    CUDA_ARCH= ;;                      # CPU partition: no CUDA
esac
MUFFTTO_DIR=$HOME/thesis/muFFTTO              # your clone (branch StefanusMichael_Dev)
VENV=$HOME/venvs/muffto-$P
echo "== partition $P | host $(hostname) | CUDA arch ${CUDA_ARCH:-none} | $(date -Is)"

module purge
module load lang/python/3.13.5-gcccore-14.3.0 mpi/openmpi/5.0.8-gcc-14.3.0 devel/cmake/3.31.8-gcccore-14.3.0
if [ -n "$CUDA_ARCH" ]; then
    module load system/cuda/12.9.1
    export CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-${CUDA_ROOT:-}}}
    nvidia-smi -L
    which nvcc
fi
module list 2>&1
which python mpicc
export CMAKE_BUILD_PARALLEL_LEVEL=${SLURM_CPUS_PER_TASK:-16}

# 1. venv and plain dependencies (incl. the ones NuMPI and muGrid need: array-api-compat, primefac, muTimer)
mkdir -p "$HOME/venvs"
python -m venv "$VENV"
source "$VENV/bin/activate"
pip install -U pip wheel
pip install numpy scipy matplotlib jax "array-api-compat>=1.12" primefac muTimer

# 2. mpi4py against the cluster's OpenMPI (the same MPI as muGrid below)
pip install --no-binary mpi4py --no-cache-dir mpi4py

# 3. CuPy (GPU partitions only)
GPU_DEFS=
if [ -n "$CUDA_ARCH" ]; then
    pip install cupy-cuda12x
    GPU_DEFS="-Ccmake.define.MUGRID_ENABLE_CUDA=ON -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"
fi

# 4. muGrid 1.6.1 from source (MPI; CUDA for the GPU of this partition)
CC=mpicc CXX=mpicxx pip install -v --no-deps --no-cache-dir --no-binary muGrid muGrid==1.6.1 $GPU_DEFS \
    > "$VENV/mugrid_build.log" 2>&1 || { tail -40 "$VENV/mugrid_build.log"; exit 1; }

# 5. NuMPI with the preconditioned trust region, and muFFTTO (editable: a git pull updates it)
pip install --no-deps "NuMPI @ git+https://github.com/muSpectre/NuMPI@feat/tr_precond"
pip install --no-deps -e "$MUFFTTO_DIR"

# 6. smoke test: every value must be True (GPU lines only on GPU partitions)
python - << 'PY'
import inspect, muGrid, NuMPI
from NuMPI.Optimization import tr_newton_bounded
from muFFTTO import domain, direct_solvers, sparse_assembly, solvers_nonlinear
print('muGrid', muGrid.__version__, '| MPI', muGrid.has_mpi)
print('NuMPI', NuMPI.__version__, '| precond option:', 'precond' in inspect.signature(tr_newton_bounded).parameters)
print('muFFTTO Discretization has device option:', 'device' in inspect.signature(domain.Discretization.__init__).parameters)
if muGrid.has_cuda:
    import cupy
    print('muGrid CUDA', muGrid.has_cuda, '| GPU visible', muGrid.is_gpu_available())
    print('CuPy sees', cupy.cuda.runtime.getDeviceCount(), 'GPU(s):', cupy.cuda.runtime.getDeviceCount() > 0)
PY
echo "== done $P $(date -Is): venv $VENV"

#!/bin/bash
# One-time build of a GPU (A100) Python environment for the TR_SRI finite-strain script on bwUniCluster 3.0.
# Run it ON a GPU node (the CUDA build needs nvcc and is compiled for the A100, CUDA architecture 80):
#     sbatch bwuc3_build_gpu_env.sh
# Result: the venv $VENV with numpy, scipy, matplotlib, jax (CPU), mpi4py (built against the cluster's OpenMPI),
#         CuPy (CUDA 12), muGrid 1.6.1 (CUDA + MPI), NuMPI feat/tr_precond and muFFTTO (editable).
# Adapted from muFFTTO/experiments/dual_preconditioner/topology_optimization/nemo_build_gpu_env.sh (NEMO2).
#SBATCH --job-name=build_muffto_gpu
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=build_muffto_gpu_%j.out
set -eo pipefail   # no -u: the Lmod module scripts use unset variables

# ---- CHECK THESE THREE NAMES ON THE CLUSTER FIRST (module avail devel/python mpi/openmpi devel/cuda) ----------------
PY_MODULE=devel/python/3.13.1-gnu-14.2     # a Python 3.11-3.13 module
MPI_MODULE=mpi/openmpi/5.0-gnu-14.2        # an OpenMPI built with the same GNU compiler as Python
CUDA_MODULE=CHANGE_ME                      # a CUDA 12.x module, e.g. devel/cuda/12.<x> (CuPy below is for CUDA 12)
# ---- paths ----------------------------------------------------------------------------------------------------------
VENV=$HOME/venv-muffto-gpu
MUFFTTO_DIR=$HOME/muFFTTO                  # your muFFTTO checkout (tested: branch StefanusMichael_Dev_rebased, b0b7768)

if [ "$CUDA_MODULE" = CHANGE_ME ]; then echo "set CUDA_MODULE first (module avail devel/cuda)"; exit 1; fi
module purge
module load "$PY_MODULE" "$MPI_MODULE" "$CUDA_MODULE"
module list 2>&1
nvidia-smi -L
which nvcc mpicc python
export CMAKE_BUILD_PARALLEL_LEVEL=${SLURM_CPUS_PER_TASK:-16}

# 1. venv and plain dependencies
python -m venv "$VENV"
source "$VENV/bin/activate"
pip install -U pip wheel
pip install numpy scipy matplotlib jax

# 2. mpi4py against the cluster's OpenMPI (same MPI as muGrid below)
pip install --no-binary mpi4py --no-cache-dir mpi4py

# 3. CuPy for CUDA 12
pip install cupy-cuda12x

# 4. muGrid 1.6.1 with CUDA (A100 = architecture 80) and MPI, compiled from source
CC=mpicc CXX=mpicxx pip install -v --no-deps --no-cache-dir --no-binary muGrid muGrid==1.6.1 \
    -Ccmake.define.MUGRID_ENABLE_CUDA=ON -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=80 \
    > "$VENV/mugrid_build.log" 2>&1 || { tail -40 "$VENV/mugrid_build.log"; exit 1; }

# 5. NuMPI with the preconditioned trust region, and muFFTTO (editable, no dependency changes)
pip install --no-deps "NuMPI @ git+https://github.com/muSpectre/NuMPI@feat/tr_precond"
pip install --no-deps -e "$MUFFTTO_DIR"

# 6. smoke test: every line must end in True
python - << 'PY'
import inspect, cupy, muGrid, NuMPI
from NuMPI.Optimization import tr_newton_bounded
from muFFTTO import domain
print('muGrid', muGrid.__version__, '| MPI', muGrid.has_mpi, '| CUDA', muGrid.has_cuda, '| GPU visible', muGrid.is_gpu_available())
print('CuPy sees', cupy.cuda.runtime.getDeviceCount(), 'GPU(s):', cupy.cuda.runtime.getDeviceCount() > 0)
print('NuMPI', NuMPI.__version__, '| precond option:', 'precond' in inspect.signature(tr_newton_bounded).parameters)
print('muFFTTO Discretization has device option:', 'device' in inspect.signature(domain.Discretization.__init__).parameters)
PY
echo "done: source $VENV/bin/activate"

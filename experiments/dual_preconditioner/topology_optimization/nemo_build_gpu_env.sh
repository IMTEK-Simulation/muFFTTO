#!/bin/bash
# Build the muFFTTO GPU environment for ONE NEMO2 GPU partition, on a node of that partition (every partition has
# its own EasyBuild stack, compiled for its CPU: a venv built on one partition dies with 'Illegal instruction' on
# another). Result:
#   ~/local/<partition>                      PnetCDF 1.14.1 (built with that partition's OpenMPI 5.0.8)
#   ~/software/muFFTTO/.venv-gpu-<partition> numpy, scipy, netCDF4, matplotlib, jax (CPU), mpi4py (from source),
#                                            CuPy (CUDA 12 / ROCm), muGrid from ~/software/muGrid (GPU arch of
#                                            the partition, -march=native), NuMPI + muFFTTO editable
# ~/software/muffto_env_gpu.sh selects it by $SLURM_JOB_PARTITION.
#   sbatch -p <partition> --gres=gpu:1 nemo_build_gpu_env.sh
#SBATCH --job-name=build_gpu_env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work/classic/fr_ml1145-muffto/runs/envbuild/build_%x_%j.out
set -eo pipefail   # no -u: the module (Lmod) scripts use unset variables
P=${PARTITION:-$SLURM_JOB_PARTITION}
case $P in
    l40s)   BACKEND=cuda; ARCH=89 ;;
    h200)   BACKEND=cuda; ARCH=90 ;;
    rtx)    BACKEND=cuda; ARCH=120 ;;
    mi300a) BACKEND=hip;  ARCH=gfx942 ;;
    *) echo "unknown partition $P"; exit 1 ;;
esac
echo "== partition $P host $(hostname) backend $BACKEND arch $ARCH $(date -Is)"
module purge
module load lang/python/3.13.5-gcccore-14.3.0 mpi/openmpi/5.0.8-gcc-14.3.0 devel/cmake/3.31.8-gcccore-14.3.0
# ROCm: the module must match the node's driver stack (/opt/rocm = 6.4.1; with 7.2.4 libamdhip64 needs newer HSA symbols)
if [ $BACKEND = cuda ]; then module load system/cuda/12.9.1; else module load system/rocm/${ROCM_VERSION:-6.4.1}; fi
module list 2>&1
export MAKEFLAGS=-j${SLURM_CPUS_PER_TASK:-16} CMAKE_BUILD_PARALLEL_LEVEL=${SLURM_CPUS_PER_TASK:-16}

# 1. PnetCDF for this partition's OpenMPI (fresh source tree: the one in ~/software is configured in place)
PREFIX=$HOME/local/$P
if [ ! -f $PREFIX/lib/libpnetcdf.so ]; then
    SRC=$(mktemp -d /tmp/pnetcdf_$P.XXXX)
    tar xzf $HOME/software/pnetcdf-1.14.1.tar.gz -C $SRC
    cd $SRC/pnetcdf-1.14.1
    ./configure --prefix=$PREFIX --enable-shared --disable-fortran CC=mpicc CXX=mpicxx > configure.log 2>&1
    make > make.log 2>&1 && make install > install.log 2>&1
    cd $HOME && rm -rf $SRC
fi
export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-} LD_LIBRARY_PATH=$PREFIX/lib:${LD_LIBRARY_PATH:-}
echo "== PnetCDF $(pkg-config --modversion pnetcdf) in $PREFIX"

# 2. venv with the Python dependencies
VENV=$HOME/software/muFFTTO/.venv-gpu-$P
python -m venv $VENV
source $VENV/bin/activate
pip install -q -U pip wheel
pip install -q numpy scipy netCDF4 matplotlib "jax" primefac array-api-compat pytest
pip install -q --no-binary mpi4py --no-cache-dir mpi4py

# 3. CuPy
if [ $BACKEND = cuda ]; then
    pip install -q cupy-cuda12x
else
    export ROCM_HOME=${ROCM_PATH:-/opt/rocm} HCC_AMDGPU_TARGET=$ARCH CUPY_INSTALL_USE_HIP=1
    ok=0
    for w in cupy-rocm-6-4 cupy-rocm-6-3 cupy-rocm-6-2 cupy-rocm-6-0; do pip install -q $w 2>/dev/null && { ok=1; echo "CuPy wheel $w"; break; }; done
    [ $ok = 1 ] || pip install -q --no-cache-dir --no-binary cupy cupy
fi

# 4. muGrid (GPU backend of the partition, CPU code for this node), NuMPI and muFFTTO (editable, shared checkouts)
if [ $BACKEND = cuda ]; then
    GPU_DEFS="-Ccmake.define.MUGRID_ENABLE_CUDA=ON -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=$ARCH"
else
    GPU_DEFS="-Ccmake.define.MUGRID_ENABLE_HIP=ON -Ccmake.define.CMAKE_HIP_ARCHITECTURES=$ARCH"
fi
pip install -v --no-cache-dir $HOME/software/muGrid $GPU_DEFS \
    -Ccmake.define.MUGRID_NATIVE_ARCH=ON > $VENV/mugrid_build.log 2>&1 \
    || { tail -40 $VENV/mugrid_build.log; exit 1; }
pip install -q -e $HOME/software/NuMPI
pip install -q --no-deps -e $HOME/software/muFFTTO

# 5. smoke test
export DEVICE=gpu MPLBACKEND=Agg JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
[ $BACKEND = cuda ] && export CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-${CUDA_ROOT:-}}}
python -c "import muGrid, cupy, NuMPI, muFFTTO; print('muGrid', muGrid.__version__, 'cuda', getattr(muGrid, 'has_cuda', '?'),
      'rocm', getattr(muGrid, 'has_rocm', '?'), 'mpi', getattr(muGrid, 'has_mpi', '?'),
      'gpu_aware_mpi', getattr(muGrid._muGrid.Communicator, 'gpu_aware_mpi', '?')); print('cupy', cupy.__version__,
      cupy.cuda.runtime.getDeviceCount(), 'device(s)')"
cd $HOME/software/muFFTTO/experiments/dual_preconditioner/topology_optimization
RUN_SUFFIX=_envtest_$P NPIX=16 MAXITER=2 ETA=0.25 SOFT_EXP=4 CG_MODE=energy GTOL=0 \
    python -u example_3D_elasticity_TO.py Green_Jacobi 2>&1 | grep -E "^[0-9]+ +[0-9]|Data saved|Error|error" | tail -5
echo "== done $P $(date -Is)"

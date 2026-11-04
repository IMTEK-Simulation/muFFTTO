# GPU environment for muFFTTO on NEMO2:  source ~/software/muffto_env_gpu.sh
# Every GPU partition has its own software stack (compiled for its CPU), so each has its own venv, built on a node of
# that partition by experiments/dual_preconditioner/topology_optimization/nemo_build_gpu_env.sh:
#   l40s   -> .venv-gpu (original), CUDA, PnetCDF in ~/local
#   rtx, h200 -> .venv-gpu-<partition>, CUDA; mi300a -> .venv-gpu-mi300a, ROCm 6.4.1; PnetCDF in ~/local/<partition>
# Several GPUs: one MPI rank per GPU, srun --gpus-per-task=1 (and INIT_NOISE_GRID for split-independent TO starts).
# The partition is taken from $SLURM_JOB_PARTITION (l40s outside a job; GPU code cannot run on the login node anyway).
_P=${SLURM_JOB_PARTITION:-l40s}
module load lang/python/3.13.5-gcccore-14.3.0
module load mpi/openmpi/5.0.8-gcc-14.3.0
if [ "$_P" = mi300a ]; then
    module load system/rocm/6.4.1            # must match the node's driver stack (/opt/rocm); 7.2.4 fails to load
else
    module load system/cuda/12.9.1
    # UCX with its CUDA transports: Open MPI reports itself CUDA-aware, so muGrid hands it device buffers; with the
    # plain lib/ucx they are memcpy'd as host memory -> segfault on more than one GPU (2026-11-04)
    module load lib/ucx-cuda/1.19.0-gcccore-14.3.0-cuda-12.9.1
fi
if [ "$_P" != l40s ] && [ -d $HOME/software/muFFTTO/.venv-gpu-$_P ]; then
    export LD_LIBRARY_PATH=$HOME/local/$_P/lib:$LD_LIBRARY_PATH
    source $HOME/software/muFFTTO/.venv-gpu-$_P/bin/activate
else
    [ "$_P" != l40s ] && echo "muffto_env_gpu.sh: no venv for partition $_P, using the l40s one" >&2
    export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
    source $HOME/software/muFFTTO/.venv-gpu/bin/activate
fi
export MUFFTTO_WS=$(ws_find muffto)
export MPLBACKEND=Agg JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
unset _P

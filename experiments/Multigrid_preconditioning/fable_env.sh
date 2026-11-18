# Environment of the large-3D-TO study (source it):  source fable_env.sh
#   laptop: micromamba env muffto-gpu;  NEMO (inside a Slurm job): the venv of the job's partition
if [ -n "$SLURM_JOB_ID" ]; then
  source ~/software/muffto_env_gpu.sh
  export PY=python CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-$CUDA_ROOT}}
else
  export PY=$HOME/micromamba/envs/muffto-gpu/bin/python CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux
fi
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 MPLBACKEND=Agg \
       PYTHONWARNINGS="ignore::RuntimeWarning"

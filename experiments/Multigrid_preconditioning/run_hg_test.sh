#!/usr/bin/env bash
# One-point hexahedra with muFFTTO's hourglass control (element trilinear_hexahedron_1Q_hg, eps = fraction of the fully
# integrated hourglass stiffness, exact per mode) vs 8-point: MG only (user), static 3D, ch_s12, laptop GPU,
# eta/h 0 / 1 / 2, eps in {0.03, 0.1, 0.3, 1}; peak GPU memory per run.
cd "$(dirname "$0")"
PY=$HOME/micromamba/envs/muffto-gpu/bin/python
export CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
       OMP_NUM_THREADS=1 PYTHONWARNINGS="ignore::RuntimeWarning"
OUT=exp_data/scaling3d_1Q; mkdir -p $OUT logs/1Q
for N in 32 64; do for EPS in 0.03 0.1 0.3 1; do for eta in 0 1 2; do
  M=logs/1Q/mem_lib_hg${EPS}_N${N}_eta${eta}.txt
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; sleep 0.2; done ) > $M & MP=$!
  $PY scaling_case_3d.py --geometry ch_s12 --grid $N --eta $eta --element trilinear_hexahedron_1Q_hg --hg-eps $EPS \
      --methods mg --maxiter 3000 --device gpu --out $OUT >> logs/1Q/run_lib_hg${EPS}_N${N}.log 2>&1
  kill $MP
  echo "1Q_hg(lib) eps $EPS N $N eta $eta peak $(sort -n $M | tail -1) MiB" >> logs/1Q/peak_memory_lib_hg.txt
done; done; done
echo done

#!/usr/bin/env bash
# Under-integrated 3D elements (trilinear_hexahedron_1Q) vs full integration: static 3D solves on the new CH geometry
# ch_s12, laptop GPU, eta/h in {0, 1, 2}, MG only (user: fast preconditioner only); peak GPU memory per (element, N, eta).
cd "$(dirname "$0")"
PY=$HOME/micromamba/envs/muffto-gpu/bin/python
export CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
       OMP_NUM_THREADS=1 PYTHONWARNINGS="ignore::RuntimeWarning"
OUT=exp_data/scaling3d_1Q; mkdir -p $OUT logs/1Q
for N in 32 64; do for E in trilinear_hexahedron trilinear_hexahedron_1Q; do for eta in 0 1 2; do
  M=logs/1Q/mem_${E}_N${N}_eta${eta}.txt
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; sleep 0.2; done ) > $M & MP=$!
  $PY scaling_case_3d.py --geometry ch_s12 --grid $N --eta $eta --element $E --methods mg --device gpu --out $OUT >> logs/1Q/run_${E}_N${N}.log 2>&1
  kill $MP
  echo "$E N $N eta $eta peak $(sort -n $M | tail -1) MiB" >> logs/1Q/peak_memory.txt
done; done; done
echo done

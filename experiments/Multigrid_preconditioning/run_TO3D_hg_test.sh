#!/usr/bin/env bash
# Small 3D TO test (laptop GPU): one-point hexahedron with hourglass control (trilinear_hexahedron_1Q_hg, HG_EPS)
# vs the 8-point hexahedron. Same setup as the NEMO 3D TO runs except grid/contrast: 32^3, eta = 2h, contrast 1e3,
# MG + Helmholtz L-BFGS (c = 4), energy CG (delay 0), start = blurred 3D Cahn-Hilliard bi_s11 (interpolated to 32^3).
#   ./run_TO3D_hg_test.sh [HG_EPS, default 0.3]     -> logs/TO3D_hg/*.log, data/example_3D_elasticity_TO/*_mgTO3Dhg_*
cd "$(dirname "$0")"
PY=$HOME/micromamba/envs/muffto-gpu/bin/python
export CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
       OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONWARNINGS="ignore::RuntimeWarning"
EPS=${1:-0.3}; N=32; mkdir -p logs/TO3D_hg
for E in trilinear_hexahedron trilinear_hexahedron_1Q_hg; do
  TAG=$([ $E = trilinear_hexahedron ] && echo 8pt || echo 1Qhg$EPS)
  M=logs/TO3D_hg/mem_$TAG.txt
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; sleep 0.5; done ) > $M & MP=$!
  ELEMENT=$E HG_EPS=$EPS DEVICE=gpu NPIX=$N ETA=$($PY -c "print(2 / $N)") SOFT_EXP=3 PF_NORMALIZED=1 PF_WEIGHT=0.2 \
  CG_MODE=energy CG_DELAY=0 GTOL=0 MAXITER=${MAXITER:-400} HISTORY_EVERY=10 SHOW_PLOTS=0 \
  INIT_FROM=$PWD/data/CH3D/CH3D_start_bi_s11.nc DESIGN_OPT=lbfgs_helm HELM_C=4 RUN_SUFFIX=_mgTO3Dhg_$TAG \
  timeout 3h $PY -u example_3D_elasticity_TO.py MG > logs/TO3D_hg/TO3D_N${N}_$TAG.log 2>&1
  kill $MP
  echo "$TAG peak $(sort -n $M | tail -1) MiB" >> logs/TO3D_hg/peak_memory.txt
done
echo done

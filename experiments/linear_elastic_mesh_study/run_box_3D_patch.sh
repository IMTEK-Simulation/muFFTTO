#!/bin/bash
# 3D cube clamped on all six faces: the family 2D-FFT preconditioner of S with a balancing second level on edge strips
# (PATCH = 1, 2 nodes), N = 8 ... 64
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
mkdir -p exp_data/logs
for n in 64 32 16 8; do for k in 1 2; do echo "$n $k"; done; done | xargs -P "${1:-4}" -L 1 bash -c '
    systemd-run --user --wait --quiet --unit=b3p-n$1-k$2 --same-dir -p MemoryMax=16G -p RuntimeMaxSec=14400 \
        -E JAX_PLATFORMS=cpu -E OMP_NUM_THREADS=4 -E OPENBLAS_NUM_THREADS=4 -E MKL_NUM_THREADS=4 \
        -E N=$1 -E INNER=family -E INNER_TOL=1e-6 -E PATCH=$2 \
        bash -c "'"$PY"' -u box_clamp_3D.py > exp_data/logs/box3d_patch_n$1_k$2.txt 2>&1"' _
grep -h "outer" exp_data/logs/box3d_patch_*.txt

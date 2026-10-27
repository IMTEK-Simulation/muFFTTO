#!/bin/bash
# 3D cantilever, corrected Green with the clamp plane through the whole cell inverted by a 2D FFT (plane_clamp_3D.py),
# against plain Green, dense S (solid face / full plane) and plain inner CG; n_H = 4 ... 32, largest first
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
mkdir -p exp_data/logs
for n in 32 16 8 4; do for v in dense_plane dense_face cg_plane green fft_plane; do echo "$n $v"; done; done |
    xargs -P "${1:-5}" -L 1 bash -c '
    systemd-run --user --wait --quiet --unit=p3-n$1-$2 --same-dir -p MemoryMax=10G -p RuntimeMaxSec=10800 \
        -E JAX_PLATFORMS=cpu -E OMP_NUM_THREADS=2 -E OPENBLAS_NUM_THREADS=2 -E MKL_NUM_THREADS=2 \
        -E N_H=$1 -E VARIANT=$2 bash -c "'"$PY"' -u plane_clamp_3D.py > exp_data/logs/plane3d_n$1_$2.txt 2>&1"' _
grep -h "outer\|S_fft" exp_data/logs/plane3d_*.txt

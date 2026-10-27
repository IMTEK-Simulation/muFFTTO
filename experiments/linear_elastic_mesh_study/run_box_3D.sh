#!/bin/bash
# 3D cube clamped on all six faces (box_clamp_3D.py): plain Green, dense S (N <= 32), inner CG, inner PCG with the
# family 2D-FFT preconditioner, the family FFT once; N = 8 ... 64, largest first
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
mkdir -p exp_data/logs
for n in 64 32 16 8; do
    for v in "INNER=green" "INNER=cg INNER_TOL=1e-8" "INNER=family INNER_TOL=1e-6" "INNER=family INNER_TOL=1e-8" \
             "INNER=familydirect"; do echo "$n $v"; done
    [ $n -le 32 ] && echo "$n INNER=dense"
done | xargs -P "${1:-4}" -L 1 bash -c '
    n=$1; shift; name=box3d_N${n}_$(echo "$@" | tr " =" "_-")
    systemd-run --user --wait --quiet --unit=b3-$name --same-dir -p MemoryMax=20G -p RuntimeMaxSec=14400 \
        -E JAX_PLATFORMS=cpu -E OMP_NUM_THREADS=4 -E OPENBLAS_NUM_THREADS=4 -E MKL_NUM_THREADS=4 \
        -E N=$n $(printf -- "-E %s " "$@") bash -c "'"$PY"' -u box_clamp_3D.py > exp_data/logs/$name.txt 2>&1"' _
grep -h "outer\|C\^-1" exp_data/logs/box3d_*.txt

#!/bin/bash
# 2D square clamped on all four faces (box_clamp_study.py): inner PCG on S preconditioned by the local Schur complement
# on a strip of width w around the fixed lines (+ balancing on the translations); dense runs store the spectra
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
mkdir -p exp_data/logs
for n in 128 64 32; do
    echo "$n INNER=dense"
    for w in 1 2 4 8; do echo "$n INNER=strip STRIP_W=$w INNER_TOL=1e-6"; done
done | xargs -P "${1:-6}" -L 1 bash -c '
    n=$1; shift; name=boxstrip_N${n}_$(echo "$@" | tr " =" "_-")
    systemd-run --user --wait --quiet --unit=bs-$name --same-dir -p MemoryMax=6G -p RuntimeMaxSec=3600 \
        -E JAX_PLATFORMS=cpu -E OMP_NUM_THREADS=2 -E OPENBLAS_NUM_THREADS=2 -E MKL_NUM_THREADS=2 \
        -E N=$n -E CLAMP=box $(printf -- "-E %s " "$@") \
        bash -c "'"$PY"' -u box_clamp_study.py > exp_data/logs/$name.txt 2>&1"' _
grep -h "outer\|strip.bal\|kff  \|family  " exp_data/logs/boxstrip_*.txt

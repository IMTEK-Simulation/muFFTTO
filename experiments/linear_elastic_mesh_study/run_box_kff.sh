#!/bin/bash
# 2D square clamped on all four faces (box_clamp_study.py): inner PCG on S preconditioned by K_FF = P K_ref P against
# plain inner CG (same tolerance 1e-6); the dense runs store the spectra of every preconditioner (K_FF included)
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
mkdir -p exp_data/logs
for n in 256 128 64 32; do
    echo "$n INNER=dense"; echo "$n INNER=kff INNER_TOL=1e-6"; echo "$n INNER=cg INNER_TOL=1e-6"
done | xargs -P "${1:-6}" -L 1 bash -c '
    n=$1; shift; name=boxkff_N${n}_$(echo "$@" | tr " =" "_-")
    systemd-run --user --wait --quiet --unit=bk-$name --same-dir -p MemoryMax=6G -p RuntimeMaxSec=7200 \
        -E JAX_PLATFORMS=cpu -E OMP_NUM_THREADS=2 -E OPENBLAS_NUM_THREADS=2 -E MKL_NUM_THREADS=2 \
        -E N=$n -E CLAMP=box $(printf -- "-E %s " "$@") \
        bash -c "'"$PY"' -u box_clamp_study.py > exp_data/logs/$name.txt 2>&1"' _
grep -h "outer\|kff  \|family  \|none  " exp_data/logs/boxkff_*.txt

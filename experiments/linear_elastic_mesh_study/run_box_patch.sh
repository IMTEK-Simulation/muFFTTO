#!/bin/bash
# 2D square clamped on all four faces (box_clamp_study.py): the family FFT preconditioner of S with a second level on
# the patches around the crossings (additive / balancing, PATCH = 0 ... 16 nodes), N = 32 ... 256; the dense runs
# store the spectra of every preconditioner
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
mkdir -p exp_data/logs
for n in 256 128 64 32; do
    echo "$n INNER=dense"
    for k in 0 2 4 8 16; do echo "$n INNER=family INNER_TOL=1e-6 PATCH=$k PATCH_MODE=bal"; done
    for k in 2 8; do echo "$n INNER=family INNER_TOL=1e-6 PATCH=$k PATCH_MODE=add"; done
done | xargs -P "${1:-6}" -L 1 bash -c '
    n=$1; shift; name=boxpatch_N${n}_$(echo "$@" | tr " =" "_-")
    systemd-run --user --wait --quiet --unit=bp-$name --same-dir -p MemoryMax=6G -p RuntimeMaxSec=7200 \
        -E JAX_PLATFORMS=cpu -E OMP_NUM_THREADS=2 -E OPENBLAS_NUM_THREADS=2 -E MKL_NUM_THREADS=2 \
        -E N=$n -E CLAMP=box $(printf -- "-E %s " "$@") \
        bash -c "'"$PY"' -u box_clamp_study.py > exp_data/logs/$name.txt 2>&1"' _
grep -h "outer\|cond" exp_data/logs/boxpatch_*.txt

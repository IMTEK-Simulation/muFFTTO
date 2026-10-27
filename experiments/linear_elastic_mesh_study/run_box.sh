#!/bin/bash
# corrected Green with the fixed dofs on full grid lines (box_clamp_study.py): two parallel lines vs all four faces,
# capacitance system S solved dense / by plain inner CG / by inner PCG with line or family FFT preconditioners
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
mkdir -p exp_data/logs
for n in 32 64 128 256; do
    for v in "CLAMP=two INNER=dense" "CLAMP=two INNER=familydirect" "CLAMP=two INNER=cg INNER_TOL=1e-8" \
             "CLAMP=box INNER=dense" "CLAMP=box INNER=familydirect" "CLAMP=box INNER=cg INNER_TOL=1e-8" \
             "CLAMP=box INNER=line INNER_TOL=1e-6" "CLAMP=box INNER=family INNER_TOL=1e-6" \
             "CLAMP=box INNER=family INNER_TOL=1e-8" "CLAMP=box INNER=family INNER_TOL=1e-4"; do
        echo "$n $v"
    done
done | xargs -P "${1:-6}" -L 1 bash -c '
    n=$1; shift; name=box_N${n}_$(echo "$@" | tr " =" "_-")
    systemd-run --user --wait --quiet --unit=bx-$name --same-dir -p MemoryMax=6G -p RuntimeMaxSec=3600 \
        -E JAX_PLATFORMS=cpu -E OMP_NUM_THREADS=2 -E OPENBLAS_NUM_THREADS=2 -E MKL_NUM_THREADS=2 \
        -E N=$n $(printf -- "-E %s " "$@") \
        bash -c "'"$PY"' -u box_clamp_study.py > exp_data/logs/$name.txt 2>&1"' _

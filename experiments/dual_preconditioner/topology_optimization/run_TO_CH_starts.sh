#!/bin/bash
# 2D TO from different initial designs: 6 Cahn-Hilliard starts (make_TO_CH_starts.py) and the standard sine start,
# 128^2, eta = 4h = 1/32, contrast 1e4, w = 0.2 (normalised objective), Green-Jacobi, energy CG stop, up to 2000
# L-BFGS iterations (stop on xtol). All runs in parallel, one core each, on the laptop CPU.
#   ./run_TO_CH_starts.sh        logs: logs/TO_CH_starts/<label>.log, data: data/example_2D_elasticity_TO*/..._<label>
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
mkdir -p logs/TO_CH_starts
export NPIX=128 ETA=0.03125 SOFT_EXP=4 PF_NORMALIZED=1 PF_WEIGHT=0.2 CG_MODE=energy GTOL=0 MAXITER=2000 \
       HISTORY_EVERY=10 MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
for label in bi_s0 bi_s1 bi_s2 bi_s3 drop_s0 drop_s1 sine; do
    init=$([ $label = sine ] || echo "INIT_FROM=$PWD/data/TO_CH_starts/CH_start_$label.nc")
    systemd-run --user --scope -q -p MemoryMax=6G env $init RUN_SUFFIX=_start_$label \
        $PY -u example_2D_elasticity_TO.py Green_Jacobi > logs/TO_CH_starts/$label.log 2>&1 &
done
wait
grep -H "OPTIMIZER\|Elapsed\|Data saved" logs/TO_CH_starts/*.log

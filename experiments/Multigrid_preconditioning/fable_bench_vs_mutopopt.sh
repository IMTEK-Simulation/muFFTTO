#!/usr/bin/env bash
# State-solve benchmark muFFTTO vs muTopOpt on one GPU (inside a job): fable_bench_vs_mutopopt.sh <N> [N ...]
#   -> logs/fable/bench_vs_mutopopt_<partition>.txt (appended)
cd "$(dirname "$0")"; source ./fable_env.sh; source ./fable_mutopopt_env.sh
OUT=logs/fable/bench_vs_mutopopt_${SLURM_JOB_PARTITION:-local}.txt; mkdir -p logs/fable
{ echo "=== $(date -Is) $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), muTopOpt $(git -C $MUTOPOPT log --oneline -1 | cut -c1-7), muFFTTO_fable $(cat ../../.fable_commit 2>/dev/null)"; } >> $OUT
for N in "$@"; do
  for DESIGN in "" "--sharpen 0.05"; do
    if [ "${ONLY:-}" != mutopopt ]; then
      $PY fable_bench_vs_mutopopt.py --code mufftto -n $N $DESIGN 2>&1 | grep -v -i warn | tail -7 >> $OUT
      $PY fable_bench_vs_mutopopt.py --code mufftto -n $N $DESIGN --operator fused 2>&1 | grep -v -i warn | tail -7 >> $OUT
      $PY fable_bench_vs_mutopopt.py --code mufftto -n $N $DESIGN --operator fused --mg-precision single 2>&1 | grep -v -i warn | tail -7 >> $OUT
    fi
    [ "${ONLY:-}" = mufftto ] && continue
    $PY fable_bench_vs_mutopopt.py --code mutopopt -n $N $DESIGN 2>&1 | grep -v -i warn | tail -7 >> $OUT
    $PY fable_bench_vs_mutopopt.py --code mutopopt -n $N $DESIGN --precision single 2>&1 | grep -v -i warn | tail -7 >> $OUT
  done
done

#!/usr/bin/env bash
# Does muTopOpt run in this environment? (inside a job) -- dependencies, version, a small solve benchmark
cd "$(dirname "$0")"; source ./fable_env.sh; source ./fable_mutopopt_env.sh
for M in muTimer tomli NuMPI scipy netCDF4; do python -c "import $M; print('$M', getattr($M, '__version__', 'ok'))" 2>&1 | tail -1; done
python -c "import muTimer" 2>/dev/null || pip install -q --target $MUTOPOPT/_deps muTimer 2>&1 | tail -2
cd $MUTOPOPT && git log --oneline -1
python -c "import muTopOpt, muGrid; print('muTopOpt', muTopOpt.__version__, 'muGrid', muGrid.version_string())" 2>&1 | tail -2
python benchmarks/solve_bench.py -n 64 --element q1 --precision double --rtol 1e-6 --solves 6 2>&1 | grep -v -i warn | tail -25

#!/usr/bin/env bash
# Code of this worktree -> my NEMO clone ~/software/muFFTTO_fable (never data, logs or figures; never ~/software/muFFTTO)
cd "$(dirname "$0")/../.."
rsync -a --exclude .git --exclude __pycache__ --exclude '.pytest_cache' --exclude 'experiments/*/logs' \
      --exclude 'experiments/*/figures' --exclude 'experiments/*/exp_data' --exclude 'experiments/dual_preconditioner' \
      --exclude 'experiments/Multigrid_preconditioning/data/example_*' ./ nemo:software/muFFTTO_fable/

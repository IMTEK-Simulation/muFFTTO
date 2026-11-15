#!/usr/bin/env bash
# Stopping-test prototype in 2D (stop_criterion_2D.py): 128^2, energy objective, nu* -0.5, for a target the design
# can nearly reach (scale 0.3) and one it cannot (scale 1) -> logs/stop_criterion/*.txt, *.npz
cd "$(dirname "$0")"; source ./fable_env.sh
mkdir -p logs/stop_criterion
for S in 0.3 1; do
  NPIX=${NPIX:-128} TARGET_NU=-0.5 TARGET_SCALE=$S MAXITER=${MAXITER:-600} $PY -u stop_criterion_2D.py > logs/stop_criterion/run_s$S.log 2>&1 &
done
wait

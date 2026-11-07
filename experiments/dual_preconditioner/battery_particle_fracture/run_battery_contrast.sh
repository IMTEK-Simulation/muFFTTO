#!/bin/bash
# Matrix-contrast study at 64^2: ./run_battery_contrast.sh <matrix stiffness> <preconditioners...>
# (waits for the log named in WAIT_FOR, if set, to finish first; at most 3 processes overall)
cd "$(dirname "$0")"
MS=$1; shift
if [ -n "$WAIT_FOR" ]; then until grep -q -E "Elapsed|Traceback" "$WAIT_FOR" 2>/dev/null; do sleep 20; done; fi
for P in "$@"; do
  MATRIX_STIFFNESS=$MS GRIDS=64 ./run_battery_grids.sh $P
done

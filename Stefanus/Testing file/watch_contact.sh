#!/bin/bash
# wait until a 64 px third-medium run reaches contact (min J in the voids < 0.1), then make and open its video
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
G=64x110_cells1x2
RUNS="${G}_TM0.001_sri_c10_L100_TM1e-3 ${G}_TM0.0001_sri_c10_L100_TM1e-4"
while true; do
  for R in $RUNS; do
    if $PY contact_monitor.py exp_data/frames_TR_X_$R > exp_data/contact_monitor_$R.txt; then
      echo "contact in $R:"; grep contact exp_data/contact_monitor_$R.txt | head -1
      others=$(for o in $RUNS ${G}_sri_c10_L100_holes; do [ "$o" != "$R" ] && echo -n "$o "; done)
      MPLBACKEND=Agg $PY make_TR_video.py X $R $others
      xdg-open figures/finite_strain_TR_X_$R.mp4 >/dev/null 2>&1 &
      exit 0
    fi
  done
  sleep 120
done

#!/bin/bash
# overnight queue. $1: PID of the 12.5 cm damage run with self-weight, $2: PID of the run without self-weight.
# Each run is stopped at its deadline if it has not finished (NetCDF and checkpoint keep everything up to then).
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export H=0.125 JAX_PLATFORMS=cpu CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux XLA_PYTHON_CLIENT_PREALLOCATE=false
wait_or_stop() {  # PID, deadline (date string)
    local end=$(date -d "$2" +%s)
    while kill -0 "$1" 2>/dev/null; do
        if [ "$(date +%s)" -ge "$end" ]; then kill "$1"; echo "$(date) stopped PID $1 at the deadline $2"; sleep 10; break; fi
        sleep 30
    done
}
wait_or_stop "$1" "06:00"
echo "$(date) main run done"
MAXITER=5000 PRECONDS=green,green_c,gj,gj_c,dual_eps_c $PY -u house_precond_study.py > exp_data/logs/house_precond_h0.125.txt 2>&1
PLOT=1 PRECONDS=green,green_c,gj,gj_c,dual_eps_c $PY house_precond_study.py >> exp_data/logs/house_precond_h0.125.txt 2>&1
echo "$(date) precond study done"
$PY house_damage_video.py --workers 12 > exp_data/logs/house_video_h0.125.txt 2>&1
echo "$(date) video done"
wait_or_stop "$2" "07:30"
echo "$(date) no-gravity run done"
GRAVITY=0 $PY house_damage_video.py --workers 12 > exp_data/logs/house_video_h0.125_nograv.txt 2>&1
echo "$(date) all done"

#!/bin/bash
# Laptop-side feeder of the NEMO battery mesh study: keeps at most MAX_L40S (12) battery jobs queued on l40s and MAX_H200
# (2) on h200 (pending + running, all job names battery_*), submitting the next entries of the ordered list below
# (small grids first, 1024^2 and 256^3 last), polling every POLL s (600). Every submitted job is appended to
# /work/classic/fr_ml1145-muffto/runs/battery/jobs.txt. Uses ssh -o BatchMode=yes nemo (the user's ControlMaster);
# stops if ssh fails. The list position is kept in $STATE so the feeder can be restarted.
#   nohup ./nemo_queue_battery.sh > logs/nemo_queue_battery.log 2>&1 &
cd "$(dirname "$0")"
MAX_L40S=${MAX_L40S:-12}; MAX_H200=${MAX_H200:-2}; POLL=${POLL:-600}
STATE=${STATE:-logs/nemo_queue_battery.state}
# entries: partition mem DIM N preconditioner
LIST=(
  "l40s 16G 2 32 Green" "l40s 16G 2 32 Green_Jacobi" "l40s 16G 2 32 dual"
  "l40s 16G 2 64 Green" "l40s 16G 2 64 Green_Jacobi" "l40s 16G 2 64 dual"
  "l40s 16G 2 128 Green" "l40s 16G 2 128 Green_Jacobi" "l40s 16G 2 128 dual"
  "l40s 24G 2 256 Green" "l40s 24G 2 256 Green_Jacobi" "l40s 24G 2 256 dual"
  "l40s 24G 3 32 Green" "l40s 24G 3 32 Green_Jacobi" "l40s 24G 3 32 dual"
  "l40s 32G 3 64 Green" "l40s 32G 3 64 Green_Jacobi" "l40s 32G 3 64 dual"
  "l40s 32G 2 512 Green" "l40s 32G 2 512 Green_Jacobi" "l40s 32G 2 512 dual"
  "l40s 64G 3 128 dual" "l40s 64G 3 128 Green_Jacobi" "l40s 64G 3 128 Green"
  "h200 200G 3 256 dual" "h200 200G 3 256 Green_Jacobi"
  "l40s 64G 2 1024 dual" "l40s 64G 2 1024 Green_Jacobi" "l40s 64G 2 1024 Green"
  "h200 200G 3 256 Green"
)
i=$(cat "$STATE" 2>/dev/null || echo 0)
REMOTE=software/muFFTTO/experiments/dual_preconditioner/battery_particle_fracture
while [ "$i" -lt "${#LIST[@]}" ]; do
  counts=$(timeout 120 ssh -o BatchMode=yes nemo 'squeue -u $USER -h -o "%P %j" | grep " battery_" | grep -v battery_smoke | awk "{print \$1}" | sort | uniq -c') \
    || { echo "$(date -Is) ssh failed: stop"; exit 1; }
  nl=$(echo "$counts" | awk '$2=="l40s"{print $1}'); nl=${nl:-0}
  nh=$(echo "$counts" | awk '$2=="h200"{print $1}'); nh=${nh:-0}
  while [ "$i" -lt "${#LIST[@]}" ]; do
    read -r part mem dim n p <<< "${LIST[$i]}"
    if [ "$part" = l40s ] && [ "$nl" -ge "$MAX_L40S" ]; then break; fi
    if [ "$part" = h200 ] && [ "$nh" -ge "$MAX_H200" ]; then break; fi
    name=battery_${dim}D_${n}_${p}
    J=$(timeout 120 ssh -o BatchMode=yes nemo "cd $REMOTE && sbatch --parsable -p $part --mem=$mem --job-name=$name sbatch_battery_gpu.sh $dim $n $p") \
      || { echo "$(date -Is) submit of $name failed: stop"; exit 1; }
    timeout 60 ssh -o BatchMode=yes nemo "echo '$J $name $part mem=$mem submitted $(date -Is)' >> /work/classic/fr_ml1145-muffto/runs/battery/jobs.txt"
    echo "$(date -Is) submitted $J $name ($part)"
    i=$((i + 1)); echo "$i" > "$STATE"
    if [ "$part" = l40s ]; then nl=$((nl + 1)); else nh=$((nh + 1)); fi
  done
  [ "$i" -lt "${#LIST[@]}" ] && sleep "$POLL"
done
echo "$(date -Is) all ${#LIST[@]} jobs submitted"

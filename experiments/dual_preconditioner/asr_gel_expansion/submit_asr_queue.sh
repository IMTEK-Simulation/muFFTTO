#!/bin/bash
# Laptop-side queue feeder for the NEMO ASR mesh study: submits the runs of QUEUE in order, keeping at most
# MAX_L40S (12) of my ASR_ jobs pending/running on l40s and MAX_H200 (2) on h200. Polls every POLL s (600).
# State: submitted entries are appended to asr_submitted.txt (laptop) and to runs/asr/jobs.txt (NEMO).
# The h200 (3D 256^3) entries wait until all 3D 128^3 runs have logged >= 3 load steps.
cd "$(dirname "$0")"
MAX_L40S=${MAX_L40S:-12}; MAX_H200=${MAX_H200:-2}; POLL=${POLL:-600}
QUEUE=()
for N in 32 64 128 256; do for P in Green Green_Jacobi dual; do QUEUE+=("l40s 2 $N $P"); done; done
for P in Green Green_Jacobi dual; do QUEUE+=("l40s 2 512 $P"); done
for N in 32 64 128; do for P in Green Green_Jacobi dual; do QUEUE+=("l40s 3 $N $P"); done; done
for P in dual Green Green_Jacobi; do QUEUE+=("l40s 2 1024 $P"); done
for P in dual Green Green_Jacobi; do QUEUE+=("h200 3 256 $P"); done
touch asr_submitted.txt
R=/work/classic/fr_ml1145-muffto/runs/asr
while true; do
  for entry in "${QUEUE[@]}"; do
    grep -qxF "$entry" asr_submitted.txt && continue
    read -r part D N P <<< "$entry"
    counts=$(ssh -o BatchMode=yes nemo "squeue -u \$USER -h -o '%P %j' | awk '\$2 ~ /^ASR_/ {print \$1}' | sort | uniq -c" ) || { echo "$(date +%H:%M) ssh failed"; break; }
    n=$(echo "$counts" | awk -v p="$part" '$2 == p {print $1}'); n=${n:-0}
    max=$MAX_L40S; [ "$part" = h200 ] && max=$MAX_H200
    [ "$n" -ge "$max" ] && continue
    if [ "$part" = h200 ]; then
      ok=$(ssh -o BatchMode=yes nemo "for p in Green Green_Jacobi dual; do f=$R/logs/ASRmesh_3D_N128_\${p}_l0.015625_cheap.log; [ -f \$f ] && grep -c '^step' \$f || echo 0; done" | awk '$1 >= 3' | wc -l)
      [ "$ok" -lt 3 ] && continue
      opts="-p h200 --mem=200G"
    else
      opts=""
    fi
    j=$(ssh -o BatchMode=yes nemo "cd ~/software/muFFTTO/experiments/dual_preconditioner/asr_gel_expansion && sbatch --parsable $opts --job-name=ASR_${D}D_${N}_${P} sbatch_asr_gpu.sh $D $N $P") || continue
    echo "$entry" >> asr_submitted.txt
    ssh -o BatchMode=yes nemo "echo '$j $part ${D}D N=$N $P submitted $(date -Is)' >> $R/jobs.txt"
    echo "$(date +%H:%M) submitted $j: $entry"
  done
  left=0; for entry in "${QUEUE[@]}"; do grep -qxF "$entry" asr_submitted.txt || left=$((left + 1)); done
  [ "$left" -eq 0 ] && { echo "$(date +%H:%M) all submitted"; exit 0; }
  sleep "$POLL"
done

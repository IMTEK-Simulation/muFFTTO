#!/bin/bash
# BOUNDED AT1 study on NEMO2 (sbatch_AT1_bounded.sh): the same grid as submit_AT1_study.sh (B2 nine pores, l/h = 8, 4, 2,
# 1, 1/2, 1/4 with l = LH / N, plus the fixed l = 1/32 series; l > 1/16 skipped; Green and dual, Green-Jacobi up to 256^2),
# but only the GPU part N >= 256: N <= 128 and the cases in SKIP were run on the laptop (run_AT1_bounded_laptop*.sh).
# QOS: at most 10 jobs per user (running + pending). This script submits while the user has fewer than MAXJOBS (8) jobs
# and writes the other cases to the queue $Q, which the jobs feed from when they end (see sbatch_AT1_bounded.sh).
# Order: Green / dual pairs from 256^2 up, Green-Jacobi (not needed for the Green / dual ratio) last.
#   DRY=1 ./submit_AT1_bounded_study.sh   lists the cases;   ./submit_AT1_bounded_study.sh  submits + writes the queue
#   job list -> $WS/runs/AT1/AT1b_jobs.txt, queue -> $WS/runs/AT1/AT1b_queue.txt
cd "$(dirname "$0")"
WS=/work/classic/fr_ml1145-muffto; JOBS=$WS/runs/AT1/AT1b_jobs.txt; Q=$WS/runs/AT1/AT1b_queue.txt
MAXJOBS=${MAXJOBS:-8}
# done (or running) on the laptop, 2026-11-13: N:l:precond
SKIP=" 256:0.03125:Green 256:0.03125:dual 256:0.015625:Green 256:0.015625:dual 256:0.0078125:dual 256:0.0078125:Green "
declare -A seen
CASES=(); GJ=()
for N in 256 512 1024; do
    for LH in 8 4 2 1 0.5 0.25 L32; do
        if [ "$LH" = L32 ]; then LS=0.03125; else LS=$(awk -v a="$LH" -v n="$N" 'BEGIN{printf "%g", a / n}'); fi
        awk -v l="$LS" 'BEGIN{exit !(l > 0.0625 + 1e-12)}' && continue          # l > 1/16: skip
        [ -n "${seen[$N,$LS]}" ] && continue; seen[$N,$LS]=1
        for P in dual Green Green_Jacobi; do
            [ "$P" = Green_Jacobi ] && [ "$N" -gt 256 ] && continue
            [[ "$SKIP" == *" $N:$LS:$P "* ]] && continue
            if [ "$P" = Green_Jacobi ]; then GJ+=("$N $LH $P"); else CASES+=("$N $LH $P"); fi
        done
    done
done
ALL=("${CASES[@]}" "${GJ[@]}")
if [ -n "$DRY" ]; then printf '%s\n' "${ALL[@]}"; exit 0; fi
: > "$Q"
for c in "${ALL[@]}"; do
    set -- $c
    if [ "$(squeue -u "$USER" -h | wc -l)" -lt "$MAXJOBS" ] && \
       J=$(sbatch --parsable -p l40s,h200,rtx,mi300a --gres=gpu:1 --time=24:00:00 --mem=16G --job-name=AT1b_N$1 \
           sbatch_AT1_bounded.sh "$1" "$2" "$3"); then
        echo "$J N=$1 l/h=$2 $3" | tee -a "$JOBS"
    else
        echo "$c" >> "$Q"
    fi
done
echo "queued (fed by the jobs): $(wc -l < "$Q")"

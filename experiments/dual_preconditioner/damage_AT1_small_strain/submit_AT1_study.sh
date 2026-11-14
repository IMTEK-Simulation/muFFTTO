#!/bin/bash
# AT1 parametric study on NEMO2 (B2 nine pores): grids 32 ... 1024, l/h = 8, 4, 2, 1, 1/2, 1/4 (l = LH / N) and the
# fixed l = 1/32 mesh-convergence series (its points coincide with the table except 512^2 and 1024^2: l/h 16, 32).
# Cases with l > 1/16 (larger than the pores) are skipped; an (N, l) pair is run once.
# Green and dual on every case, Green-Jacobi only up to 256^2. Only iterations and results are compared (no timings),
# so the jobs go wherever they start first: N <= 128 on the CPU (serial -> restartable), N >= 256 on any GPU partition.
#   DRY=1 ./submit_AT1_study.sh   lists the jobs;   ./submit_AT1_study.sh  submits; job list -> $WS/runs/AT1/AT1_jobs.txt
cd "$(dirname "$0")"
WS=/work/classic/fr_ml1145-muffto; JOBS=$WS/runs/AT1/AT1_jobs.txt
declare -A seen
submit() {   # N LH
    local N=$1 LH=$2 LS
    if [ "$LH" = L32 ]; then LS=0.03125; else LS=$(awk -v a="$LH" -v n="$N" 'BEGIN{printf "%g", a / n}'); fi
    awk -v l="$LS" 'BEGIN{exit !(l > 0.0625 + 1e-12)}' && return          # l > 1/16: skip
    [ -n "${seen[$N,$LS]}" ] && return; seen[$N,$LS]=1
    for P in dual Green Green_Jacobi; do
        [ "$P" = Green_Jacobi ] && [ "$N" -gt 256 ] && continue
        if [ "$N" -le 128 ]; then
            T=$([ "$N" -le 64 ] && echo 04:00:00 || echo 24:00:00)
            ARGS="-p cpu --time=$T --mem=8G"
        else
            ARGS="-p l40s,h200,rtx,mi300a --gres=gpu:1 --time=24:00:00 --mem=16G"
        fi
        if [ -n "$DRY" ]; then echo "N=$N l/h=$LH l=$LS $P : $ARGS"; continue; fi
        J=$(sbatch --parsable $ARGS --job-name=AT1_N${N} sbatch_AT1.sh "$N" "$LH" "$P")
        echo "$J N=$N l/h=$LH l=$LS $P $ARGS" | tee -a "$JOBS"
    done
}
for N in 32 64 128 256 512 1024; do
    for LH in 8 4 2 1 0.5 0.25 L32; do submit $N $LH; done
done

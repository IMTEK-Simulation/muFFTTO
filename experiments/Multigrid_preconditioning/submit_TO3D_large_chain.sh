#!/usr/bin/env bash
# Submit a chain of large 3D TO runs on NEMO2 (run it on the login node, in this folder of the clone):
#   ./submit_TO3D_large_chain.sh <label> <start.nc> "<N> <GPUs> <segments> <time> <host mem>" ["<N> ..." ...]
# e.g. ./submit_TO3D_large_chain.sh aux_s0.3 $PWD/data/CH3D/CH3D_start_bi_s11.nc \
#          "256 1 1 12:00:00 64G" "512 1 2 24:00:00 160G" "1024 8 2 24:00:00 900G"
# Per grid: <segments> jobs, each continuing the previous one from its checkpoint (afterany); a segment that finds
# the run finished exits at once. The first job of the next grid waits for the last segment (afterok) and starts from
# this grid's _final.nc -- if the run did not finish, that file is missing and the chain stops there.
# Settings of sbatch_TO3D_large.sh (SYMMETRY, ETA_H, MAXITER, ...) are taken from the environment.
# PARTITION (default h200). AFTER=<job id>: the first job waits for that job (afterok), e.g. the run that writes
# <start.nc>. DRY_RUN=1 prints the sbatch commands only. Note: h200 allows 8 GPUs per user in total, so a full-node
# job waits for all other h200 jobs of the user; on l40s set SBATCH_EXCLUDE=g3606,g3608 (GPU ECC errors).
# RULES: jobs > 24 h, full-node (8 GPU) production runs: ask the user first.
LABEL=$1; START=$2; shift 2
PARTITION=${PARTITION:-h200}; DEP=${AFTER:+--dependency=afterok:$AFTER}
SOFT=$(awk "BEGIN{printf \"%g\", 10 ^ -${SOFT_EXP:-3}}")
for LEVEL in "$@"; do
  set -- $LEVEL; N=$1; G=$2; SEGMENTS=$3; TIME=$4; MEM=$5
  H=$(echo $TIME | awk -F: '{print ($1 * 3600 + $2 * 60 + $3)}')
  for SEGMENT in $(seq $SEGMENTS); do
    CMD="sbatch --parsable -p $PARTITION --gres=gpu:$G --ntasks=$G --mem=$MEM --time=$TIME -J FB_TO3D_${N} $DEP sbatch_TO3D_large.sh $N $START $LABEL"
    BUDGET=${TIME_BUDGET:-$((H - 1500))}                     # stop 25 min before the limit (TIME_BUDGET overrides)
    if [ "${DRY_RUN:-0}" = 1 ]; then echo "TIME_BUDGET=$BUDGET $CMD"; JOB="<job>"
    else JOB=$(TIME_BUDGET=$BUDGET $CMD); echo "N $N segment $SEGMENT: job $JOB ($G GPU, $TIME) start $START"; fi
    DEP="--dependency=afterany:$JOB"
  done
  DEP="--dependency=afterok:$JOB"
  START=/work/classic/fr_ml1145-muffto/runs/FABLE/data/N${N}_soft${SOFT}_eta$(awk "BEGIN{printf \"%g\", ${ETA_H:-2} / $N}")_energy_${LABEL}_final.nc
done

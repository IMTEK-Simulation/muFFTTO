#!/bin/bash
# One segment of a large 3D TO run (to3d_large.py) on NEMO2, one MPI rank per GPU of one node.
#   sbatch [-p h200] --gres=gpu:<G> --ntasks=<G> [--time=..] [--mem=..] sbatch_TO3D_large.sh <N> <start.nc> <label>
#   -> log   $WS/runs/FABLE/TO3D_N<N>_<label>.log        (appended by every segment)
#      data  $WS/runs/FABLE/data/N<N>_..._<label>_{checkpoint,final,history}.nc
# A segment continues from the run's checkpoint if there is one, else starts from <start.nc> (a design on any grid,
# e.g. the _final.nc of the next coarser grid; never noise). It ends with _final.nc (converged or MAXITER), or after
# TIME_BUDGET seconds with a fresh checkpoint for the next segment. A finished run is left alone.
# Chains over segments and grids: submit_TO3D_large_chain.sh.
# Env (defaults): ETA_H=2 (interface width in voxels), SYMMETRY=none, WARM_STORE=host, MAXITER=500, CHECKPOINT_EVERY=10,
#   HISTORY_EVERY=0 (no history file; a frame is N^3 x 8 bytes), TIME_BUDGET=82800 (23 h, for a 24 h job),
#   MG_COARSE_ELEMENT, TARGET_NU=-0.5, TARGET_SCALE=0.3, SOFT_EXP=3
# GPU memory (measured, see FABLE_NOTES.md): ~900 B/voxel all-in + ~4 GiB per rank under MPI: 256^3 15 GiB,
# 512^3 114 GiB (fits one H200), 2 H200: 61 GiB each; 1024^3 on 8 H200: ~120 GiB each (estimate).
# Host memory (--mem): WARM_STORE=host needs 24 B/voxel per stored field (12 fields with 6 load cases, 4 with a
# SYMMETRY) + ~0.4 kB/voxel for L-BFGS and I/O: 512^3 ~120 GB, 1024^3 ~800 GB (6 load cases).
#SBATCH --job-name=FB_TO3D
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/work/classic/fr_ml1145-muffto/runs/FABLE/logs/%x-%j.out
N=$1; START=$2; LABEL=$3
[ -n "$LABEL" ] || { echo "usage: sbatch_TO3D_large.sh <N> <start.nc> <label>"; exit 2; }
WS=/work/classic/fr_ml1145-muffto
cd ~/software/muFFTTO_fable/experiments/Multigrid_preconditioning
source ./fable_env.sh
export DEVICE=gpu NPIX=$N ETA=$(python -c "print(${ETA_H:-2} / $N)") SOFT_EXP=${SOFT_EXP:-3} PF_WEIGHT=0.2 CG_DELAY=0 GTOL=0 \
       OBJECTIVE=energy TARGET_NU=${TARGET_NU:--0.5} TARGET_SCALE=${TARGET_SCALE:-0.3} HELM_C=4 \
       ELEMENT=trilinear_hexahedron_1Q_hg HG_EPS=0.3 SYMMETRY=${SYMMETRY:-none} WARM_STORE=${WARM_STORE:-host} \
       MAXITER=${MAXITER:-500} CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-10} HISTORY_EVERY=${HISTORY_EVERY:-0} \
       TIME_BUDGET=${TIME_BUDGET:-82800} INIT_FROM=$START OUT_DIR=$WS/runs/FABLE/data RUN_SUFFIX=_$LABEL
LOG=$WS/runs/FABLE/TO3D_N${N}_$LABEL.log
OUT=$OUT_DIR/N${N}_soft$(python -c "print(f'{10.0 ** -$SOFT_EXP:g}')")_eta$(python -c "print(f'{$ETA:g}')")_energy_$LABEL
if [ ! -f "$START" ] && [ ! -f "${OUT}_checkpoint.nc" ] && [ ! -f "${OUT}_final.nc" ]; then
  echo "start design $START does not exist (and the run has no checkpoint): nothing to start from" | tee -a "$LOG"; exit 1
fi
{ echo "=== job $SLURM_JOB_ID $(hostname) $(date -Is): N $N, $SLURM_NTASKS GPU(s), start $START, label $LABEL"; nvidia-smi -L; } >> "$LOG"
srun --ntasks=$SLURM_NTASKS --gpus-per-task=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python -u to3d_large.py >> "$LOG" 2>&1
RC=$?
echo "=== job $SLURM_JOB_ID done $(date -Is) exit $RC" >> "$LOG"
exit $RC

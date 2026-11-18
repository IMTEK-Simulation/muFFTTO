#!/bin/bash
# l/h sweep of the multigrid_vs_dual study: ONE grid N and ONE l/h per job, all six elastic-solver configurations one
# after another on the same L40S GPU (timings comparable within the job):
#   dual, MG V(1,1), MG V(2,2)  x  CG energy test d_min 10 and 0      (MG: 3 warm power iterations per setup)
#   sbatch --job-name=MGL_<N>_lh<lh> sbatch_lh_sweep.sh <N> <lh>
# Problem: polydisperse pores (1e-3), k = 1e-4, l = lh / N, Gc = 0.016 l (fixed strength: Gc / l of the 64^2, l = 2h,
# Gc = 5e-4 run), uniaxial strain xx to 0.12 in 120 steps (increment 1e-3), stop 3 steps after cut-through; no NetCDF
# fields (OUTPUT_NC=0: final damage as .npy). A configuration whose log already has 'Elapsed time' is skipped, so a
# resubmitted job continues with the missing ones.
# Env: LOAD_STEPS / LOAD_MAX (smoke test), CONFIGS (subset, e.g. "dual MG11"), SUFFIX (appended)
# Output: $WS/runs/MG_vs_dual/{exp_data,logs,figures}: logs/AT2_N<N>_<config>_sw_lh<lh>.log,
#   exp_data/AT2_N<N>_<MG|dual>_polydisperse_pores_l<l>_sw_lh<lh>[_v11|_v22][_d0]_iterations.csv
#SBATCH --job-name=MGL_sweep
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --exclude=g3606,g3608,g3610         # GPU ECC errors / bad nodes
#SBATCH --output=/work/classic/fr_ml1145-muffto/runs/MG_vs_dual/slurm/%x-%j.out
N=$1; LH=$2
[ -n "$LH" ] || { echo "usage: sbatch_lh_sweep.sh <N> <l/h>"; exit 2; }
WS=/work/classic/fr_ml1145-muffto
RUNS=$WS/runs/MG_vs_dual
cd ~/software/muFFTTO_fable/experiments/multigrid_vs_dual
for d in exp_data figures logs; do
  mkdir -p $RUNS/$d
  [ -L $d ] || { [ -d $d ] && rmdir $d 2>/dev/null; ln -sfn $RUNS/$d $d; }
done
source ~/software/muffto_env_gpu.sh
export DEVICE=gpu CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-$CUDA_ROOT}} XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTHONWARNINGS="ignore::RuntimeWarning"
L=$(python -c "print($LH / $N)")
export MICROSTRUCTURE=polydisperse_pores NPIX=$N LENGTH_SCALE=$L GC=$(python -c "print(0.016 * $LH / $N)") \
       LOAD_MAX=${LOAD_MAX:-0.12} LOAD_STEPS=${LOAD_STEPS:-120} CHECKPOINT_EVERY=0 OUTPUT_NC=0 MG_POWER=3
RC=0
for C in ${CONFIGS:-dual MG11 MG22 dual_d0 MG11_d0 MG22_d0}; do
  case $C in
    dual*) P=dual; TAG= ;;
    MG11*) P=MG; TAG=_v11; export MG_DEGREE=1 ;;
    MG22*) P=MG; TAG=_v22; export MG_DEGREE=2 ;;
  esac
  case $C in *_d0) D=0; DT=_d0 ;; *) D=10; DT= ;; esac
  LOG=logs/AT2_N${N}_${C}_sw_lh${LH}${SUFFIX:-}.log
  if [ -f $LOG ] && grep -q "Elapsed time" $LOG; then echo "$C done, skipped"; continue; fi
  { echo "=== job $SLURM_JOB_ID $(hostname) $(date -Is): N $N, l/h $LH (l $L, Gc $GC), $C"; nvidia-smi -L; } > $LOG
  CG_DELAY=$D RUN_SUFFIX=_sw_lh${LH}${TAG}${DT}${SUFFIX:-} python -u example_2D_phase_field_fracture_AT2.py $P >> $LOG 2>&1
  rc=$?; [ $rc -ne 0 ] && RC=$rc
  echo "=== $C done $(date -Is) exit $rc" >> $LOG
done
exit $RC

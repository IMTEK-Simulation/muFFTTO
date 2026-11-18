#!/bin/bash
# One AT2 run of the multigrid_vs_dual study on one L40S GPU (NEMO2):
#   sbatch [--export=ALL,VAR=..] sbatch_mg_vs_dual.sh <N> <precond>        precond: MG | dual | Green
# Problem: polydisperse pores (contrast 1e-3), l/h = 2 (LENGTH_SCALE = 2 / N), k = 1e-4, uniaxial strain xx up to
# 0.08 in 80 steps, stop 3 steps after the cell is cut through; CG energy test rtol 1e-6, d_min CG_DELAY (default 10).
# Env passed through: CG_DELAY, MG_POWER, MG_DEGREE, LOAD_STEPS (smoke test), SUFFIX (appended to the run name)
# Output: $WS/runs/MG_vs_dual/{exp_data,figures,logs} (symlinked from the study folder);
#   log logs/AT2_N<N>_<P>_lh2<SUFFIX>.log, CSV exp_data/AT2_N<N>_<P>_polydisperse_pores_l<l>_lh2<SUFFIX>_iterations.csv
#SBATCH --job-name=MGD_AT2
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --exclude=g3606,g3608,g3610         # GPU ECC errors / bad nodes
#SBATCH --output=/work/classic/fr_ml1145-muffto/runs/MG_vs_dual/slurm/%x-%j.out
N=$1; P=$2
[ -n "$P" ] || { echo "usage: sbatch_mg_vs_dual.sh <N> <MG|dual|Green>"; exit 2; }
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
export MICROSTRUCTURE=polydisperse_pores NPIX=$N LENGTH_SCALE=$(python -c "print(2.0 / $N)") \
       RUN_SUFFIX=_lh2${SUFFIX:-} CHECKPOINT_EVERY=0
LOG=logs/AT2_N${N}_${P}_lh2${SUFFIX:-}.log
{ echo "=== job $SLURM_JOB_ID $(hostname) $(date -Is): N $N, $P, CG_DELAY ${CG_DELAY:-10}, MG_POWER ${MG_POWER:-15}"
  nvidia-smi -L; } > $LOG
python -u example_2D_phase_field_fracture_AT2.py $P >> $LOG 2>&1
RC=$?
echo "=== job $SLURM_JOB_ID done $(date -Is) exit $RC" >> $LOG
exit $RC

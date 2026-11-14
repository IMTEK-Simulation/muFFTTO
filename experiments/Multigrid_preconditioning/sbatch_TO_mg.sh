#!/bin/bash
# 2D TO with multigrid vs FFT preconditioners on one L40S (NEMO2), NEMO production settings of sbatch_TO2D_start_gpu.sh
# (contrast 1e4, PF_NORMALIZED=1, PF_WEIGHT=0.2, energy CG with warm starts, eta = ETA_H h, up to MAXITER L-BFGS iterations,
# xtol stop) with CG_DELAY=0, started from a Cahn-Hilliard design (INIT_FROM, interpolated to N).
#   sbatch sbatch_TO_mg.sh <MG|Green|Green_Jacobi|dual> <N> [start .nc, default data/CH_start_bi_s0.nc]
#   -> $WS/runs/MG_TO/TO_N<N>_<P>_<label>_eta<ETA_H>h.log; data in data/example_2D_elasticity_TO/ (..._mgTO_<label>)
# Env: ETA_H (interface width in pixels, default 4; the user's rule: eta >= 2-4 h), LABEL (default from the start file
#      name), MAXITER (default 2000), CG_DELAY (default 0), DESIGN_OPT (default lbfgs = NuMPI; lbfgs_helm, ncg, ncg_helm:
#      design_optimizers.py, tagged _<DESIGN_OPT> in the log and data names), HELM_C (Helmholtz strength, default
#      1 = untagged; other values tagged _c<HELM_C>), SCRIPT (default example_2D_elasticity_TO.py;
#      example_3D_elasticity_TO.py -> 3D, logs TO3D_...)
#SBATCH --job-name=MG_TO
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=g3606,g3608         # GPU ECC errors
#SBATCH --output=/work/classic/fr_ml1145-muffto/logs/MG_TO-%j.out
P=$1; N=$2
cd ~/software/muFFTTO/experiments/Multigrid_preconditioning
INIT=${3:-$PWD/data/CH_start_bi_s0.nc}
LABEL=${LABEL:-$(basename "$INIT" .nc | sed 's/^CH_start_//')}
WS=/work/classic/fr_ml1145-muffto
source ~/software/muffto_env_gpu.sh
export DEVICE=gpu CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-$CUDA_ROOT}} XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTHONWARNINGS="ignore::RuntimeWarning:muFFTTO.discretization_library"
mkdir -p $WS/runs/MG_TO
ETA_H=${ETA_H:-4}
DESIGN_OPT=${DESIGN_OPT:-lbfgs}; OPT_TAG=$([ "$DESIGN_OPT" = lbfgs ] || echo "_$DESIGN_OPT")
HELM_C=${HELM_C:-1}; [ "$HELM_C" = 1 ] || OPT_TAG="${OPT_TAG}_c${HELM_C}"
SCRIPT=${SCRIPT:-example_2D_elasticity_TO.py}; PFX=$(case $SCRIPT in *3D*) echo TO3D;; *) echo TO;; esac)
LOG=$WS/runs/MG_TO/${PFX}_N${N}_${P}_${LABEL}_eta${ETA_H}h${OPT_TAG}${LOG_SUFFIX:-}.log
{ echo "job $SLURM_JOB_ID $(hostname) $(date -Is) P $P N $N init $INIT"; nvidia-smi -L; } >> "$LOG"
NPIX=$N ETA=$(python -c "print($ETA_H / $N)") SOFT_EXP=4 PF_NORMALIZED=1 PF_WEIGHT=0.2 CG_MODE=energy CG_DELAY=${CG_DELAY:-0} \
GTOL=0 MAXITER=${MAXITER:-2000} HISTORY_EVERY=10 SHOW_PLOTS=0 INIT_FROM=$INIT DESIGN_OPT=$DESIGN_OPT HELM_C=$HELM_C \
RUN_SUFFIX=_mgTO_${LABEL}${OPT_TAG}${LOG_SUFFIX:-} \
python -u $SCRIPT $P >> "$LOG" 2>&1
echo "WALL done $(date -Is) exit $?" >> "$LOG"

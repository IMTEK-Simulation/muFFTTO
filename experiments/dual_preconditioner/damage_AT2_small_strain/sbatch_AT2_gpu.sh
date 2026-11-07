#!/bin/bash
# One AT2 run on one L40S GPU (NEMO2), settings of gypsum_rve/run_AT2_gypsum_rve_large_gpu.sh (the laptop queue):
#   sbatch sbatch_AT2_gpu.sh gypsum <N> grains|pores <precond>     -> gypsum_rve/logs/AT2_N<N>_<P>_polydisperse_pores<tag>_l0.0195915_gypsum.log
#   sbatch sbatch_AT2_gpu.sh chgc2  <N> -            <precond>     -> the same with phase 2 cracking (PHASE2_GC=1.7407)
#   sbatch sbatch_AT2_gpu.sh ch     <N> -            <precond>     -> schneider2025_notch/logs/AT2_N<N>_<P>_incl0.1_l0.03125_k0.0001_ch.log
#   sbatch sbatch_AT2_gpu.sh ch3D   <N> -            <precond>     -> 3D Cahn-Hilliard (geometry/cahn_hilliard_3D_128.npy), ch settings
#                                                                     with l = 1/16 (fixed: l/h = 2 / 4 / 8 at 32^3 / 64^3 / 128^3)
#   [LS=<l>] sbatch sbatch_AT2_gpu.sh grains3D <N> - <precond>    -> 3D stiff tough grains (polydisperse balls, 10x), defaults of
#                                                                     run_AT2_3D_fixed_l.sh, l = 1/16 (logs/AT2_N<N>_<P>_..._incl10_l0.0625_3D.log)
# Options (environment, via sbatch --export=ALL,...):
#   LOAD_STEPS=<n>            override (smoke test)
#   EXTRA_SUFFIX=<s>          appended to RUN_SUFFIX and the log name (smoke test: _smoketest)
#   RESTART_FROM=<.npz/.nc>   continue a run, with SEG=<k> (output and log get the suffix _r<k>)
#   OUTPUT_FIELDS=damage      store only the damage field per load step (256^3: ~27 GB instead of ~320 GB per run)
# 48 h limit: the run is stopped after BUDGET s (default 47 h 30 min) and the job resubmits itself (SEG+1) with
# RESTART_FROM = this segment's <output>_checkpoint.npz (CHECKPOINT_EVERY staggered iterations), else its .nc
# (last converged load step); at most MAXSEG (default 4) segments
# Output: <case>/exp_data, <case>/figures -> symlinks into $MUFFTTO_WS/runs/AT2/<case>/
#SBATCH --job-name=AT2_gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --exclude=g3606,g3608         # GPU ECC errors
#SBATCH --output=/work/classic/fr_ml1145-muffto/logs/AT2_gpu-%j.out
set -e
CASE=$1; N=$2; S=$3; P=$4
SEG=${SEG:-0}; BUDGET=${BUDGET:-171000}; MAXSEG=${MAXSEG:-4}
BASE_EXTRA=${EXTRA_SUFFIX:-}
EXTRA_SUFFIX=$BASE_EXTRA; [ "$SEG" -gt 0 ] && EXTRA_SUFFIX=${EXTRA_SUFFIX}_r$SEG || true
source ~/software/muffto_env_gpu.sh
export DEVICE=gpu CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-$CUDA_ROOT}} XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu       # as in the laptop queue (jax on the CPU)
cd ~/software/muFFTTO/experiments/dual_preconditioner/damage_AT2_small_strain
case $CASE in
  gypsum)
    cd gypsum_rve
    case $S in
      pores)  export HOLES=1 HOLE_STIFFNESS=1e-4; tag=_holes0.0001 ;;
      grains) export INCLUSION_STIFFNESS=10;      tag=_incl10 ;;
      *) echo "setting must be grains or pores"; exit 2 ;;
    esac
    export MICROSTRUCTURE=polydisperse_pores LENGTH_SCALE=0.0195915 GC=8.4635e-6 LAME=0.277778,0.416667 \
           LOAD_MAX=0.015 LOAD_STEPS=${LOAD_STEPS:-150} RUN_SUFFIX=_gypsum${EXTRA_SUFFIX:-}
    LOG=logs/AT2_N${N}_${P}_polydisperse_pores${tag}_l0.0195915_gypsum${EXTRA_SUFFIX:-}.log ;;
  ch)
    cd schneider2025_notch
    export GEOMETRY_FILE=geometry/cahn_hilliard_512.npy INCLUSION_STIFFNESS=0.1 LENGTH_SCALE=0.03125 GC=1.2857e-8 \
           LAME=0.576923,0.384615 LOAD_DIR=yy LOAD_MAX=2e-3 LOAD_STEPS=${LOAD_STEPS:-200} \
           RUN_SUFFIX=_ch${EXTRA_SUFFIX:-} K_RESIDUAL=1e-4
    LOG=logs/AT2_N${N}_${P}_incl0.1_l0.03125_k0.0001_ch${EXTRA_SUFFIX:-}.log ;;
  chgc2)   # Cahn-Hilliard with a crackable phase 2, Gc_2 = 1.7407 Gc (PHASE2_GC, run_AT2_cahn_hilliard_gc2.sh)
    cd schneider2025_notch
    export GEOMETRY_FILE=geometry/cahn_hilliard_512.npy INCLUSION_STIFFNESS=0.1 PHASE2_GC=1.7407 LENGTH_SCALE=0.03125 \
           GC=1.2857e-8 LAME=0.576923,0.384615 LOAD_DIR=yy LOAD_MAX=2e-3 LOAD_STEPS=${LOAD_STEPS:-200} \
           RUN_SUFFIX=_ch${EXTRA_SUFFIX:-} K_RESIDUAL=1e-4
    LOG=logs/AT2_N${N}_${P}_incl0.1_gc1.7407_l0.03125_k0.0001_ch${EXTRA_SUFFIX:-}.log ;;
  ch3D)    # 3D confirmation of the 2D Cahn-Hilliard scaling (fixed l, grid refinement)
    cd schneider2025_notch
    export DIM=3 GEOMETRY_FILE=geometry/cahn_hilliard_3D_128.npy INCLUSION_STIFFNESS=0.1 LENGTH_SCALE=0.0625 GC=1.2857e-8 \
           LAME=0.576923,0.384615 LOAD_DIR=yy LOAD_MAX=2e-3 LOAD_STEPS=${LOAD_STEPS:-200} \
           RUN_SUFFIX=_ch${EXTRA_SUFFIX:-} K_RESIDUAL=1e-4
    LOG=logs/AT2_N${N}_${P}_incl0.1_l0.0625_3D_k0.0001_ch${EXTRA_SUFFIX:-}.log ;;
  grains3D)   # 3D stiff tough grains, l = LS (env, default 1/16; fixed l/h series: LS = 4/N) (32^3 done locally 2026-10-31)
    LS=${LS:-0.0625}
    export DIM=3 MICROSTRUCTURE=polydisperse_pores INCLUSION_STIFFNESS=10 LENGTH_SCALE=$LS \
           ${LOAD_STEPS:+LOAD_STEPS=$LOAD_STEPS} RUN_SUFFIX=${EXTRA_SUFFIX:-}
    LOG=logs/AT2_N${N}_${P}_polydisperse_pores_incl10_l${LS}_3D${EXTRA_SUFFIX:-}.log ;;
  *) echo "case must be gypsum, ch, chgc2, ch3D or grains3D"; exit 2 ;;
esac
export NPIX=$N
echo "job $SLURM_JOB_ID host $(hostname) start $(date -Is) case=$CASE N=$N $S $P log=$PWD/$LOG RESTART_FROM=${RESTART_FROM:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# GPU use of this run, every 10 min (and once after 2 min) -> slurm .out
( sleep 120; while true; do echo "$(date +%H:%M) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ') util $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)"; sleep 600; done ) &
mon=$!
start=$(date +%s)
rc=0
timeout -s TERM -k 120 "$BUDGET" python -u example_2D_phase_field_fracture_AT2.py "$P" > "$LOG" 2>&1 || rc=$?
kill $mon 2>/dev/null || true
echo "done $(date -Is) exit $rc wall $(( $(date +%s) - start )) s"
STEM=exp_data/$(basename "$LOG" .log)          # the log name is the output name
ls -la ${STEM}* 2>/dev/null || true
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  if [ "$SEG" -ge $((MAXSEG - 1)) ]; then echo "time limit, MAXSEG=$MAXSEG reached: not continuing"; exit 1; fi
  if [ -f "${STEM}_checkpoint.npz" ]; then FROM=$PWD/${STEM}_checkpoint.npz
  elif [ -f "${STEM}.nc" ]; then FROM=$PWD/${STEM}.nc
  else echo "time limit, no checkpoint or .nc: not continuing"; exit 1; fi
  cd ..
  NEXT=$(sbatch --parsable -p "$SLURM_JOB_PARTITION" --mem="${SLURM_MEM_PER_NODE}M" --job-name="${SLURM_JOB_NAME}_r$((SEG + 1))" \
         --export=ALL,SEG=$((SEG + 1)),RESTART_FROM=$FROM,EXTRA_SUFFIX=$BASE_EXTRA sbatch_AT2_gpu.sh "$CASE" "$N" "$S" "$P")
  echo "time limit: continued as job $NEXT (segment $((SEG + 1)), RESTART_FROM=$FROM)"
  exit 0
fi
exit $rc

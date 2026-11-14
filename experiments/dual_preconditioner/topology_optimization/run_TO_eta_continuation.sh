#!/bin/bash
# 2D eta-continuation test (sine start): wide interface on a coarse grid first, then sharpen and refine.
# Settings of run_TO_CH_starts.sh (contrast 1e4, w = 0.2 normalised, a = 1, energy CG stop, stop on xtol, up to 2000
# L-BFGS iterations), so its 128^2 eta = 1/32 runs (sine: closed pockets?, Cahn-Hilliard starts) are the baselines.
#   stage 1   64^2  eta = 1/8  (8h)  sine start                          Green-Jacobi          suffix _cont_s1
#   chain U  128^2  eta = 1/32 (4h)  from stage 1 (eta / 4 in one jump)  GJ, dual, Green       suffix _cont_U
#   chain S   64^2  eta = 1/16 (4h)  from stage 1                        Green-Jacobi          suffix _cont_s2
#            128^2  eta = 1/32 (4h)  from it (eta / 2 per stage)         GJ, dual, Green       suffix _cont_S
#   chain C   64^2  eta = 1/8 from the smooth Cahn-Hilliard start bi_s0 (data/TO_CH_starts, rho = (c+1)/2, ~53 % grey)
#            128^2  eta = 1/32 from it                                    Green-Jacobi          suffix _cont_C
# The last stage runs with all three preconditioners from the same start (cost comparison on a sharper design).
#   ./run_TO_eta_continuation.sh      logs: logs/TO_eta_continuation/, data: data/example_2D_elasticity_TO/
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
L=logs/TO_eta_continuation; D=$PWD/data/example_2D_elasticity_TO
mkdir -p $L
export SOFT_EXP=4 PF_NORMALIZED=1 PF_WEIGHT=0.2 CG_MODE=energy GTOL=0 MAXITER=2000 HISTORY_EVERY=10 SHOW_PLOTS=0 \
       MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
run() {  # <log> <precond> env...
    local log=$1 P=$2; shift 2
    systemd-run --user --scope -q -p MemoryMax=6G env "$@" $PY -u example_2D_elasticity_TO.py $P > $L/$log.log 2>&1
}
final() { echo "$D/N$1_soft0.0001_energy-warm_$2_eta_$3_pfw0.2_l_bfgs$4_final.nc"; }

( run C1_N64_eta8h Green_Jacobi NPIX=64 ETA=0.125 INIT_FROM=$PWD/data/TO_CH_starts/CH_start_bi_s0.nc RUN_SUFFIX=_cont_C1
  C1=$(final 64 Green_Jacobi 0.125 _cont_C1); [ -f "$C1" ] || { echo "chain C stage 1 failed"; exit 1; }
  run C_N128_Green_Jacobi Green_Jacobi NPIX=128 ETA=0.03125 INIT_FROM=$C1 RUN_SUFFIX=_cont_C ) &
run s1_N64_eta8h Green_Jacobi NPIX=64 ETA=0.125 RUN_SUFFIX=_cont_s1
S1=$(final 64 Green_Jacobi 0.125 _cont_s1); [ -f "$S1" ] || { echo "stage 1 failed"; exit 1; }
for P in Green_Jacobi dual Green; do
    run U_N128_$P $P NPIX=128 ETA=0.03125 INIT_FROM=$S1 RUN_SUFFIX=_cont_U &
done
( run s2_N64_eta4h Green_Jacobi NPIX=64 ETA=0.0625 INIT_FROM=$S1 RUN_SUFFIX=_cont_s2
  S2=$(final 64 Green_Jacobi 0.0625 _cont_s2); [ -f "$S2" ] || { echo "stage S2 failed"; exit 1; }
  for P in Green_Jacobi dual Green; do
      run S_N128_$P $P NPIX=128 ETA=0.03125 INIT_FROM=$S2 RUN_SUFFIX=_cont_S &
  done; wait ) &
wait
grep -H "OPTIMIZER\|Elapsed\|Data saved" $L/*.log

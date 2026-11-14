#!/usr/bin/env bash
# Peak GPU memory of the PROCESS (nvidia-smi --query-compute-apps: other GPU jobs do not count) for
#  TO:     3D TO with the setup of run_TO3D_hg_test.sh (6 load cases, MG + Helmholtz L-BFGS c4, eta 2h, contrast 1e3,
#          bi_s11 3D CH start), MAXITER=2
#  static: one MG-CG solve (scaling_case_3d.py, ch_s12, eta 2h, contrast 1e4)
# for trilinear_hexahedron_1Q_hg (eps 0.3) at 32/64/128^3 and trilinear_hexahedron at 32/64^3 (128^3: > 8 GB laptop GPU).
# -> logs/mem_hg/peak.txt  (MiB per process, CUDA context included)
cd "$(dirname "$0")"
PY=$HOME/micromamba/envs/muffto-gpu/bin/python
export CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
       OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONWARNINGS="ignore::RuntimeWarning"
mkdir -p logs/mem_hg
OUT=logs/mem_hg/peak.txt

run_and_sample() {   # run_and_sample <label> <log> <command...>: sample the command's own GPU memory every 0.3 s
  local label=$1 log=$2; shift 2
  "$@" > "$log" 2>&1 & local P=$! M=0 U
  while kill -0 $P 2>/dev/null; do
    U=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$P '$1==p {print $2}')
    [ -n "$U" ] && [ "$U" -gt "$M" ] && M=$U
    sleep 0.3
  done
  wait $P; local rc=$?
  echo "$label peak $M MiB exit $rc" >> $OUT
}

for spec in "trilinear_hexahedron_1Q_hg 32" "trilinear_hexahedron 32" "trilinear_hexahedron_1Q_hg 64" \
            "trilinear_hexahedron 64" "trilinear_hexahedron_1Q_hg 128"; do
  set -- $spec; E=$1; N=$2
  run_and_sample "static $E N $N" logs/mem_hg/static_${E}_N$N.log \
    $PY scaling_case_3d.py --geometry ch_s12 --grid $N --eta 2 --element $E --hg-eps 0.3 --methods mg --device gpu \
        --out /tmp/claude-1000/-home-martin-PycharmProjects-muFFTTO-experiments-Multigrid-preconditioning/e27ddd28-e897-4160-ae9b-1d8fb92e339f/scratchpad/memtest
  run_and_sample "TO     $E N $N" logs/mem_hg/TO_${E}_N$N.log \
    env ELEMENT=$E HG_EPS=0.3 DEVICE=gpu NPIX=$N ETA=$($PY -c "print(2 / $N)") SOFT_EXP=3 PF_NORMALIZED=1 PF_WEIGHT=0.2 \
        CG_MODE=energy CG_DELAY=0 GTOL=0 MAXITER=2 HISTORY_EVERY=10 SHOW_PLOTS=0 \
        INIT_FROM=$PWD/data/CH3D/CH3D_start_bi_s11.nc DESIGN_OPT=lbfgs_helm HELM_C=4 RUN_SUFFIX=_memtest \
        $PY -u example_3D_elasticity_TO.py MG
done
echo done >> $OUT

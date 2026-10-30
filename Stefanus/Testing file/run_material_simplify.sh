#!/bin/bash
# can the material be simplified? true holes, 32 px, 1x2 cells, SRI Q1, TR Newton + Green, X compression to eps 0.03
# (through buckling at ~0.016): inclusion contrast {10, 100} x Lambda/mu {10, 100}, and the paper's (1000, 1000)
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
COMMON="-n 32 -inc 6 --eps_max 0.03 --direction X --cells 1x2 --imperfection 1e-3 --element bilinear_rectangle --sri --precond green"
{
echo "--incl_contrast 1000 --lam_over_mu 1000 --tag _c1000_L1000"
for C in 10 100; do
  for L in 10 100; do
    echo "--incl_contrast $C --lam_over_mu $L --tag _c${C}_L${L}"
  done
done
} | xargs -P 5 -I{} sh -c "systemd-run --user --scope -p MemoryMax=6G -q env OMP_NUM_THREADS=2 MPLBACKEND=Agg \
    JAX_PLATFORMS=cpu timeout 4h $PY stefanus_2D_finite_strain_NeoHookean_TR.py $COMMON {} > exp_data/log_simplify\$(echo '{}' | grep -o '_c[0-9]*_L[0-9]*').txt 2>&1"

#!/bin/bash
# The two-phase microstructure test of Schneider & Kaestner (FFEMS 2025, Sec. 4.3, Fig. 12-14), small strain, with our
# own Cahn-Hilliard geometry (make_cahn_hilliard_geometry.py -> geometry/cahn_hilliard_512.npy; their exact image is not
# available): 1 x 1 m periodic cell, phase 1 E = 210 GPa (matrix, cracks), phase 2 E = 21 GPa (stiffness 0.1); here
# phase 2 is tough (no crack driving force, INCLUSION_STIFFNESS=0.1) instead of their Gc = 4.7 vs 2.7 N/mm -- the paper
# finds the cracks in phase 1. Gc = 2.7 N/mm (unit E x 1 m), nu = 0.3 (true plane strain), l = 1/32 (31 mm; paper 6 mm,
# not resolvable on 64^2 / 128^2), uniaxial strain E_yy to 2e-3 in 200 steps (stop once cut through).
# k_res = 1e-6, 1e-5, 1e-4 (their Fig. 14: <= 5e-6 did not finish in 24 h) x Green / Green-Jacobi / dual x 64^2, 128^2.
# See README.md; paper: Schneider_Kaestner_2025_FFEMS_FFT_phase_field_fracture.pdf
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu GEOMETRY_FILE=geometry/cahn_hilliard_512.npy INCLUSION_STIFFNESS=0.1 \
  LENGTH_SCALE=0.03125 GC=1.2857e-8 LAME=0.576923,0.384615 LOAD_DIR=yy LOAD_MAX=2e-3 LOAD_STEPS=200 RUN_SUFFIX=_ch
job() {  # N K P
    k=$(python3 -c "print(f'{float(\"$2\"):g}')")
    systemd-run --user --scope -p MemoryMax=4G -q env NPIX=$1 K_RESIDUAL=$2 \
        $PY -u example_2D_phase_field_fracture_AT2.py $3 > logs/AT2_N$1_$3_incl0.1_l0.03125_k${k}_ch.log 2>&1
    echo "$(date +%H:%M:%S) done N=$1 k=$2 $3"
}
export -f job; export PY
for N in 128 64; do for K in 1e-6 1e-5 1e-4; do for P in Green Green_Jacobi dual; do echo "$N $K $P"; done; done; done \
    | xargs -P 12 -L 1 bash -c 'job $0 $1 $2'

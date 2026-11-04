#!/bin/bash
# Single notch under tension of Schneider & Kaestner (FFEMS 2025, Sec. 4.1.1), small strain: periodic 50 x 50 mm cell
# (length unit 50 mm, stress unit E), notch x = 5 ... 15 mm at mid height (one pixel row, stiffness k_res, damage
# solved), l = 1 mm, Gc = 2.7 N/mm, E = 210 GPa, nu = 0.3 (true plane strain Lame), uniaxial strain E_yy up to 1.1e-3
# in 100 steps. Residual stiffness k_res = 1e-6, 1e-5, 1e-4 (their Fig. 14: k_res <= 5e-6 did not finish in 24 h with
# their Green + NGMRES solver) x Green / Green-Jacobi / dual x 64^2, 128^2; 12 runs at a time; suffix _notch
# (see README.md; paper: Schneider_Kaestner_2025_FFEMS_FFT_phase_field_fracture.pdf)
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu LENGTH_SCALE=0.02 GC=2.5714e-7 LAME=0.576923,0.384615 \
  NOTCH=0.1,0.3,0.5 LOAD_DIR=yy LOAD_MAX=1.1e-3 LOAD_STEPS=100 RUN_SUFFIX=_notch
job() {  # N K P
    systemd-run --user --scope -p MemoryMax=4G -q env NPIX=$1 PORE_CONTRAST=$2 K_RESIDUAL=$2 \
        $PY -u example_2D_phase_field_fracture_AT2.py $3 > logs/AT2_N$1_$3_l0.02_k$(python3 -c "print(f'{float(\"$2\"):g}')")_notch.log 2>&1
    echo "$(date +%H:%M:%S) done N=$1 k=$2 $3"
}
export -f job; export PY
for N in 128 64; do for K in 1e-6 1e-5 1e-4; do for P in Green Green_Jacobi dual; do echo "$N $K $P"; done; done; done \
    | xargs -P 12 -L 1 bash -c 'job $0 $1 $2'

#!/bin/bash
# Practical application with a physical length scale: the 3D-printed gypsum of the Damage Mechanics Challenge
# (Navidtehrani, Duddu, Martinez-Paneda, EFM 301 (2024) 110046): E = 600 MPa, nu = 0.2, Gc = 0.13 N/mm,
# sigma_c = 4.05 MPa -> l = 27 E Gc / (256 sigma_c^2) = 0.5 mm (AT2). Periodic 25.6 x 25.6 mm RVE (length unit 25.6 mm,
# stress unit E) with the nine polydisperse disks (radii 1.3 - 3.6 mm): porous gypsum (pores of stiffness k, no damage
# in the pores) and gypsum with stiff tough grains (10x). Fixed physical l = 0.5 mm: l/h = 1.25 / 2.5 / 5 / 10 on
# 64^2 / 128^2 / 256^2 / 512^2 (l/h >= 5 for mesh objectivity, Kristensen et al. 2021). Plane strain, uniaxial strain
# E_xx to 0.015 in 150 steps. Here: 64^2 - 256^2, Green / Green-Jacobi / dual, 12 runs at a time. See README.md
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export MICROSTRUCTURE=polydisperse_pores MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu LENGTH_SCALE=0.0195915 \
  GC=8.4635e-6 LAME=0.277778,0.416667 LOAD_MAX=0.015 LOAD_STEPS=150 RUN_SUFFIX=_gypsum
job() {  # N SETTING P
    case $2 in
        pores) envs="HOLES=1 HOLE_STIFFNESS=1e-4"; tag=_holes0.0001 ;;
        grains) envs="INCLUSION_STIFFNESS=10";     tag=_incl10 ;;
    esac
    systemd-run --user --scope -p MemoryMax=4G -q env NPIX=$1 $envs \
        $PY -u example_2D_phase_field_fracture_AT2.py $3 > logs/AT2_N$1_$3_polydisperse_pores${tag}_l0.0195915_gypsum.log 2>&1
    echo "$(date +%H:%M:%S) done N=$1 $2 $3"
}
export -f job; export PY
for N in 256 128 64; do for S in pores grains; do for P in Green Green_Jacobi dual; do echo "$N $S $P"; done; done; done \
    | xargs -P 12 -L 1 bash -c 'job $0 $1 $2'

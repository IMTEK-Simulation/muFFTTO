#!/bin/bash
# The rest of the AT1 study grid (submit_AT1_study.sh) with the BOUNDED AT1, on the laptop, after run_AT1_bounded_laptop.sh:
# grids 32 ... 1024, l/h = 8, 4, 2, 1, 1/2, 1/4 (l = LH / N) and l = 1/32; l > 1/16 skipped; Green and dual everywhere,
# Green_Jacobi up to 256^2. Cases already present (iterations CSV exists) are skipped.
#   CPU: N <= 128, 12 in parallel.   GPU: N >= 256 one at a time (two runs do not fit in the 8 GB), smallest first,
#   starting after the 256^2 pair of run_AT1_bounded_laptop.sh has finished.
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
export MPLBACKEND=Agg MODEL=AT1 MICROSTRUCTURE=polydisperse_pores LOAD_MAX=0.32 LOAD_STEPS=320
mkdir -p logs
cases() {   # print "N l P" for every case of the study not yet run
    declare -A seen
    for N in 32 64 128 256 512 1024; do
        for LH in 8 4 2 1 0.5 0.25 L32; do
            if [ "$LH" = L32 ]; then LS=0.03125; else LS=$(awk -v a="$LH" -v n="$N" 'BEGIN{printf "%g", a / n}'); fi
            awk -v l="$LS" 'BEGIN{exit !(l > 0.0625 + 1e-12)}' && continue
            [ -n "${seen[$N,$LS]}" ] && continue; seen[$N,$LS]=1
            for P in dual Green Green_Jacobi; do
                [ "$P" = Green_Jacobi ] && [ "$N" -gt 256 ] && continue
                [ -e "exp_data/AT1_N${N}_${P}_polydisperse_pores_l${LS}_bounded_iterations.csv" ] && continue
                echo "$N $LS $P"
            done
        done
    done
}
run() {   # N l P device
    local N=$1 LS=$2 P=$3 DEV=$4
    if [ "$DEV" = cpu ]; then export OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu; fi
    DEVICE=$DEV NPIX=$N LENGTH_SCALE=$LS $PY -u example_2D_phase_field_fracture_AT1_bounded.py "$P" \
        > "logs/AT1b_N${N}_${P}_l${LS}.log" 2>&1
}
export -f run; export PY
ALL=$(cases)
echo "$ALL" | awk '$1 <= 128' | xargs -r -P 12 -L 1 bash -c 'run $0 $1 $2 cpu' &
(   until grep -q Elapsed logs/AT1b_N256_Green_l0.03125.log 2>/dev/null && grep -q Elapsed logs/AT1b_N256_dual_l0.03125.log 2>/dev/null; do sleep 30; done
    echo "$ALL" | awk '$1 >= 256' | sort -n -k1,1 -s | awk '{print $0, ($3 == "Green_Jacobi")}' | sort -n -k1,1 -k4,4 -s \
        | while read N LS P _; do
              [ -e "exp_data/AT1_N${N}_${P}_polydisperse_pores_l${LS}_bounded_iterations.csv" ] && continue
              run $N $LS $P gpu
          done ) &
wait
grep -H "Elapsed" logs/AT1b_*.log | wc -l

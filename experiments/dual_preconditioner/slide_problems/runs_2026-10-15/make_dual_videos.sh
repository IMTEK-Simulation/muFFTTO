#!/bin/bash
# [GRID=adapted] make_dual_videos.sh N : for every slide whose dual run (N^2, regular grid, or the adapted grid with
# GRID=adapted) is finished (and newer than its video; FORCE=1 redoes all), a video of the dual run with the Green and
# Green-Jacobi runs of the same problem (any device, when they exist) in grey -> videos_N<N>[_adapted]_dual_vs_G_GJ/
cd "$(dirname "$0")"
# Green / Green-Jacobi reference runs and the video script stay in the slides folder; the dual runs are here
SLIDES=../../../../examples/phase_field_fracture_adapted_grid/exp_data/slides
n=$1
grid=${GRID:-regular}
out=videos_N${n}$([ $grid = adapted ] && echo _adapted)_dual_vs_G_GJ; mkdir -p $out
P=~/micromamba/envs/muffto-gpu/bin/python; export JAX_PLATFORMS=cpu
declare -A tag=([1_newton_cg]=_elastic_newton-cg_lin-void [2_trust_region]=_elastic_lin-void [3_contact]=_elastic
                [4_fracture]='')
declare -A gc=([1_newton_cg]='*' [2_trust_region]='*' [3_contact]='*' [4_fracture]=0.5)
declare -A colour=([1_newton_cg]=det_F [2_trust_region]=det_F [3_contact]=det_F [4_fracture]=damage)
declare -A title=([1_newton_cg]='1. Compression, Newton-CG' [2_trust_region]='2. Compression, trust-region Newton-CG'
                  [3_contact]='3. Compression with contact (third medium), trust region'
                  [4_fracture]='4. Contact and fracture ($G_c = 0.5$), trust region')
for slide in 1_newton_cg 2_trust_region 3_contact 4_fracture; do
  # newest finished run of a preconditioner (a finished run has 'elapsed' in its log), CPU or GPU
  run() { local roots=$SLIDES/runs; [ $1 = D ] && roots=$(ls -d runs_dual_*)
          for root in $roots; do
            for r in $(ls -d $root/*_${grid}_N${n}_Gc${gc[$slide]}_$1_inner1e-06${tag[$slide]}{,_gpu}_2026* 2>/dev/null); do
              grep -q '^elapsed' $r/run.log 2>/dev/null && echo $r; done; done | sort -t_ -k1 | tail -1; }
  d=$(run D)
  [ -n "$d" ] || continue
  [ -z "$FORCE" ] && [ -f $out/$slide.mp4 ] && [ $out/$slide.mp4 -nt $d/contact_fracture.nc ] && continue
  refs=()
  g=$(run G);  [ -n "$g" ]  && refs+=(--reference "Green=$g/contact_fracture.nc")
  gj=$(run GJ); [ -n "$gj" ] && refs+=(--reference "Green-Jacobi=$gj/contact_fracture.nc")
  $P $SLIDES/../../make_slide_video.py $d/contact_fracture.nc $out/$slide --colour ${colour[$slide]} --det-f-spread 0.25 \
      --title "${title[$slide]}, ${n}²$([ $grid = adapted ] && echo ' adapted'), dual vs Green / Green-Jacobi" \
      --label 'dual GK(A⁻¹)G' "${refs[@]}" --cg-range 10 3e5 2>&1 | tail -1
done

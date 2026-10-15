#!/bin/bash
# the three studies of README: adapted grid trust region and contact, regular grid trust region (control)
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=3 MPLBACKEND=Agg
cd "$(dirname "$0")"
S=/home/martin/PycharmProjects/muFFTTO/examples/phase_field_fracture_adapted_grid/exp_data/slides
/home/martin/micromamba/envs/muffto-gpu/bin/python block_jacobi_study.py $S/N32_adapted/2_trust_region/run.py /home/martin/PycharmProjects/muFFTTO/examples/phase_field_fracture_adapted_grid/exp_data/slides/runs/bubble_array_F11-0.4_Q1_adapted_N32_Gc0.1_GJ_inner1e-06_elastic_lin-void_20261013-084704/contact_fracture.nc results_adapted_N32_trust_region --increments 10 30 50 70 90 > results_adapted_N32_trust_region.log 2>&1 &
/home/martin/micromamba/envs/muffto-gpu/bin/python block_jacobi_study.py $S/N32_adapted/3_contact/run.py /home/martin/PycharmProjects/muFFTTO/examples/phase_field_fracture_adapted_grid/exp_data/slides/runs/bubble_array_F11-0.4_Q1_adapted_N32_Gc0.1_GJ_inner1e-06_elastic_20261013-084704/contact_fracture.nc results_adapted_N32_contact --increments 10 50 80 95 > results_adapted_N32_contact.log 2>&1 &
/home/martin/micromamba/envs/muffto-gpu/bin/python block_jacobi_study.py $S/N32/2_trust_region/run.py /home/martin/PycharmProjects/muFFTTO/examples/phase_field_fracture_adapted_grid/exp_data/slides/runs/bubble_array_F11-0.4_Q1_regular_N32_Gc0.1_GJ_inner1e-06_elastic_lin-void_20261013-082307/contact_fracture.nc results_regular_N32_trust_region --increments 10 30 50 70 90 > results_regular_N32_trust_region.log 2>&1 &
wait

#!/bin/bash
# copy the Fourel-model scripts and geometries to the NEMO run folder (self-contained; library = ~/software/muFFTTO)
cd "$(dirname "$0")"
R=/work/classic/fr_ml1145-muffto/runs/asr_fourel
ssh -o BatchMode=yes nemo "mkdir -p $R/geometry $R/exp_data $R/logs $R/slurm"
rsync -a fourel_nonlocal_damage.py make_fourel_geometry_2D.py make_fourel_geometry_3D.py sbatch_fourel_gpu.sh \
      nemo:$R/
rsync -a geometry/fourel2D_s1.npz geometry/fourel3D_s1*.npz nemo:$R/geometry/
ssh -o BatchMode=yes nemo "chmod +x $R/sbatch_fourel_gpu.sh; ls -la $R $R/geometry"

"""
The two microstructures for the slide 'Green's operator preconditioning -> grid size independence': one final TO design
of threshold_designs_study.py (128^2, contrast 1e4, l = 2h, Green) as computed (smooth interphase) and thresholded at
rho = 0.5 (two phases). rho = 1 solid (black), 0 void (white). Pixels are repeated, so PDF viewers cannot blur them.
    python plot_slide_design_smooth_thresholded.py  ->  figures/slide_design_{smooth,thresholded}.{pdf,png} + talk figures/
"""
import os
import shutil

import matplotlib.pyplot as plt
import netCDF4
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
FINAL = os.path.join(HERE, 'data', 'example_2D_elasticity_TO', 'nemo_eta2h',
                     'N128_soft0.0001_energy-warm_Green_eta_0.015625_w_5.0_l_bfgs_final.nc')
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')

rho = np.asarray(netCDF4.Dataset(FINAL)['phase_field_solution'][:]).squeeze()
for name, field in (('smooth', np.clip(rho, 0, 1)), ('thresholded', (rho > 0.5).astype(float))):
    fig = plt.figure(figsize=(2.5, 2.5))
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.imshow(field.T.repeat(8, 0).repeat(8, 1), cmap='gray_r', vmin=0, vmax=1, interpolation='nearest', origin='lower')
    ax.set_xticks([]), ax.set_yticks([])
    for ext in ('pdf', 'png'):
        p = os.path.join(HERE, 'figures', f'slide_design_{name}.{ext}')
        fig.savefig(p, dpi=410)
        shutil.copy(p, TALK)
    plt.close(fig)
    print(name, field.shape, f'{np.mean((field > 0) & (field < 1)):.3f} of pixels strictly between 0 and 1')

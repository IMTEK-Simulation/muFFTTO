"""Picture for the 'Density based approach' slide: one unit cell of a converged design rho(x) (2048^2 run, contrast 1e3,
l = 4h, dual), solid (rho = 1) black, void (rho = 0) white, exact pixels.
    python plot_slide_density_design.py  ->  figures/slide_density_design.{png,pdf} (+ copy to the talk figures/)
"""
import os
import shutil

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
r = np.asarray(netCDF4.Dataset(os.path.join(HERE, 'data/example_2D_elasticity_TO/nemo/'
     'N2048_soft0.001_energy-warm_dual_eta_0.001953125_w_5.0_l_bfgs_final.nc'))['phase_field_solution'][:]).squeeze().T
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 16})
fig = plt.figure(figsize=(4.5, 5.4))
ax = fig.add_axes([0.03, 0.10, 0.94, 0.72])
ax.imshow(r, origin='lower', cmap='Greys', vmin=0, vmax=1, interpolation='nearest', extent=(0, 1, 0, 1))
ax.set_xticks([]), ax.set_yticks([])
for sp in ax.spines.values():                                   # no frame around the cell
    sp.set_visible(False)
ax.set_xlabel(r'density field $\rho(\mathbf{x})$ in unit cell $\mathcal{Y}$', labelpad=8, fontsize=24)
# labels above the cell, arrows down into the solid (rho = 1) and into a void (rho = kappa_void) near the top edge;
# target points chosen on the data
void = np.argwhere(r < 0.01)
solid = np.argwhere(r > 0.99)
n = r.shape[0]
ps = solid[np.argmin(np.hypot(solid[:, 1] / n - 0.07, solid[:, 0] / n - 0.94))]   # (row = y, col = x)
pv = void[np.argmin(np.hypot(void[:, 1] / n - 0.88, void[:, 0] / n - 0.86))]
for (row, col), lab, tx in ((ps, r'$\rho = 1$', 0.14), (pv, r'$\rho = \kappa_{\mathrm{void}}$', 0.88)):
    ax.annotate(lab, xy=(col / n, row / n), xycoords='data', xytext=(tx, 1.14), textcoords='axes fraction',
                fontsize=26, va='bottom', ha='center', annotation_clip=False,
                arrowprops=dict(arrowstyle='-|>', color='tab:red', lw=2.2, mutation_scale=18, shrinkA=2, shrinkB=2))
out = os.path.join(HERE, 'figures', 'slide_density_design')
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=300, bbox_inches='tight')
    shutil.copy(f'{out}.{ext}', TALK)
print('wrote', out)

"""
Problem sketch for the slide 'Phase-field simulation of copper sintering': the initial packing used in all runs
(geometry/particles_s2.npz: 8 Cu particles, periodic 128 nm cell, drawn on exact pixels) with the die pressure of
50 MPa along y, as in sintering_phase_field.py (Sigma = -P e_y e_y).
    python plot_sintering_problem.py  ->  figures/sintering_problem.{pdf,png} (+ copy to the talk figures/)
"""
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 15})

geo = np.load(os.path.join(HERE, 'geometry', 'particles_s2.npz'))
centres, radii, cell = geo['centres'], geo['radii'], float(geo['cell'])
n = 640
x = (np.arange(n) + 0.5) * cell / n
X, Y = np.meshgrid(x, x, indexing='ij')
dist = np.stack([np.hypot((X - c[0] + cell / 2) % cell - cell / 2, (Y - c[1] + cell / 2) % cell - cell / 2) - r
                 for c, r in zip(centres, radii)])
grain, solid = dist.argmin(0), dist.min(0) < 0
tones = np.array([0.30, 0.55, 0.42, 0.68, 0.36, 0.60, 0.48, 0.74])          # grey level of each grain, 1 = black
field = np.where(solid, tones[grain], 0.0)

fig = plt.figure(figsize=(3.6, 4.3))
ax = fig.add_axes([0.13, 0.17, 0.74, 0.74 * 3.6 / 4.3])
ax.imshow(field.T, cmap='gray_r', vmin=0, vmax=1, origin='lower', interpolation='nearest', extent=(0, cell, 0, cell))
ax.set_xticks([]), ax.set_yticks([])
for s in ax.spines.values():
    s.set_linestyle((0, (4, 3))), s.set_linewidth(1.2)

# labels on the largest particle (fully inside the cell) and in the largest pore
whole = [i for i in np.argsort(-radii) if radii[i] < centres[i][0] < cell - radii[i] and radii[i] < centres[i][1] < cell - radii[i]]
ax.text(*centres[whole[0]], 'Cu', color='white', ha='center', va='center', fontsize=17)
pore = np.unravel_index(ndimage.distance_transform_edt(np.pad(~solid, n // 2, mode='wrap'))[n // 2:-n // 2, n // 2:-n // 2].argmax(), solid.shape)
ax.text(x[pore[0]], x[pore[1]], 'pore', ha='center', va='center', fontsize=13)

# die pressure, top and bottom
for xa in np.linspace(0.14, 0.86, 5):
    for y0, y1 in ((1.15, 1.02), (-0.15, -0.02)):
        ax.annotate('', xy=(xa, y1), xytext=(xa, y0), xycoords='axes fraction', arrowprops=dict(arrowstyle='-|>', color='k', lw=1.6))
ax.text(0.5, 1.17, r'$\bar\sigma_{yy} = -50$ MPa', transform=ax.transAxes, ha='center', va='bottom')
ax.text(0.5, -0.17, 'periodic cell, 128 nm', transform=ax.transAxes, ha='center', va='top')
for ext in ('pdf', 'png'):
    p = os.path.join(HERE, 'figures', f'sintering_problem.{ext}')
    fig.savefig(p, dpi=300, facecolor='white')
    shutil.copy(p, TALK)
a = np.asarray(plt.imread(os.path.join(HERE, 'figures', 'sintering_problem.png')))[..., :3]
d = (a < 0.9).any(2)
print(a.shape, 'edge ink T/B/L/R', d[:4].sum(), d[-4:].sum(), d[:, :4].sum(), d[:, -4:].sum(), 'pore at', pore, 'solid', solid.mean())

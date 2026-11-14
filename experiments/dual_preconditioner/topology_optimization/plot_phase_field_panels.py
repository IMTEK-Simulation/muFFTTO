"""
The panels of plot_phase_field_explainer.py as separate files for the slide 'Phase field regularization'
(rho = phase field, 1 solid / 0 void; 1024^2 TO design, contrast 1e4, eta = 4h, dual):
  pf_sharp.pdf  pf_diffuse.pdf  pf_zoom.pdf  pf_profile.pdf   -> figures/ and the talk figures/
    python plot_phase_field_panels.py
"""
import os
import shutil

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from matplotlib.patches import Rectangle
from scipy import ndimage

HERE = os.path.dirname(os.path.realpath(__file__))
FINAL = os.path.join(HERE, 'data', 'example_2D_elasticity_TO', 'nemo',
                     'N1024_soft0.0001_energy-warm_dual_eta_0.00390625_w_5.0_l_bfgs_final.nc')
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})

rho = np.asarray(netCDF4.Dataset(FINAL)['phase_field_solution'][:]).squeeze()
n = rho.shape[0]
lab, _ = ndimage.label(rho < 0.5)
best = None
for sl in ndimage.find_objects(lab):
    x0, x1, y0, y1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
    if x0 > 20 and y0 > 20 and x1 < n - 20 and y1 < n - 20:
        size = max(x1 - x0, y1 - y0)
        if best is None or size > best[0]:
            best = (size, x0, x1, y0, y1)
size, x0, x1, y0, y1 = best
cx, cy, half = (x0 + x1) // 2, (y0 + y1) // 2, size // 2 + 24
win = rho[cx - half:cx + half, cy - half:cy + half].T
m = win.shape[0]
zs, col = 36, m // 2
rows = np.where((win[:, col] > 0.5)[1:] != (win[:, col] > 0.5)[:-1])[0]
zr = rows[rows > m // 2][0] if np.any(rows > m // 2) else rows[-1]
zx0, zy0 = int(np.clip(col - zs // 2, 0, m - zs)), int(np.clip(zr - zs // 2, 0, m - zs))
zoom = win[zy0:zy0 + zs, zx0:zx0 + zs]
kw = dict(cmap='gray_r', vmin=0, vmax=1, interpolation='nearest', origin='lower')


def save(fig, name):
    for ext in ('pdf', 'png'):
        p = os.path.join(HERE, 'figures', f'{name}.{ext}')
        fig.savefig(p, dpi=200, bbox_inches='tight', pad_inches=0.02)
        if os.path.isdir(TALK):
            shutil.copy(p, TALK)
    plt.close(fig)


for name, img, box in (('pf_sharp', (win > 0.5).astype(float), False), ('pf_diffuse', win, True)):
    fig, a = plt.subplots(figsize=(4, 4))
    a.imshow(img, **kw)
    if box:
        a.add_patch(Rectangle((zx0 - 0.5, zy0 - 0.5), zs, zs, fill=False, ec='tab:orange', lw=2.5))
    a.set_xticks([]), a.set_yticks([])
    save(fig, name)

fig, a = plt.subplots(figsize=(4, 4))
a.imshow(zoom, **kw)
for k in np.arange(-0.5, zs, 1):
    a.axhline(k, color='tab:blue', lw=0.4, alpha=0.6)
    a.axvline(k, color='tab:blue', lw=0.4, alpha=0.6)
a.set_xticks([]), a.set_yticks([])
for s in a.spines.values():
    s.set_color('tab:orange'), s.set_linewidth(3)
save(fig, 'pf_zoom')

fig, b = plt.subplots(figsize=(4.8, 2.9))
s = np.linspace(-6, 6, 1201)
prof = lambda x: 1.0 / (1.0 + np.exp(-x))
b.plot(s, (s > 0).astype(float), color='0.55', lw=2.5, ls='--', label='sharp')
b.plot(s, prof(s), color='k', lw=3, label='phase field')
w = np.log(9)
b.annotate('', xy=(-w, 0.5), xytext=(w, 0.5), arrowprops=dict(arrowstyle='<->', color='tab:red', lw=2))
b.text(-w - 0.25, 0.5, '$\\propto l$', color='tab:red', ha='right', va='center', fontsize=14)
b.set_xlim(-6, 6), b.set_ylim(-0.05, 1.08), b.set_xticks(range(-6, 7, 2))
b.set(xlabel='distance across the interface  $s/l$', ylabel='$\\rho$')
b.legend(loc='upper left', fontsize=12, framealpha=0.95)
b.grid(alpha=0.3)
save(fig, 'pf_profile')
print('wrote pf_sharp, pf_diffuse, pf_zoom, pf_profile (.pdf/.png) to figures/ and', TALK)

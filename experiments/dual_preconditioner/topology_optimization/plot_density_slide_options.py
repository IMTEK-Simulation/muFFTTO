"""Mock-ups for the 'Density based approach' slide picture (four options), from the 2048^2 design (contrast 1e3,
l = 4h, dual). Solid (rho = 1) black, void (rho = 0) white.
    python plot_density_slide_options.py  ->  figures/slide_density_options_preview.png
"""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.realpath(__file__))
r = np.asarray(netCDF4.Dataset(os.path.join(HERE, 'data/example_2D_elasticity_TO/nemo/'
     'N2048_soft0.001_energy-warm_dual_eta_0.001953125_w_5.0_l_bfgs_final.nc'))['phase_field_solution'][:]).squeeze().T
n = r.shape[0]
K = 1e-4
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13,
                     'text.latex.preamble': r'\usepackage{amsmath,amssymb}'})
fig = plt.figure(figsize=(16, 11), layout='constrained')
sf = fig.subfigures(2, 2, hspace=0.04, wspace=0.04)


def cell(ax, a, ext=(0, 1, 0, 1)):
    return ax.imshow(a, origin='lower', cmap='Greys', vmin=0, vmax=1, interpolation='nearest', extent=ext)


# option 1: unit cell + interpolation curve
s1 = sf[0, 0]
s1.suptitle(r'\textbf{1}\; unit cell $+$ material interpolation', fontsize=16)
a, b = s1.subplots(1, 2, gridspec_kw=dict(width_ratios=[1, 1]))
im = cell(a, r)
a.set_xticks([]), a.set_yticks([]), a.set_title(r'design $\rho(\mathbf{x})$')
cb = s1.colorbar(im, ax=a, fraction=0.046, pad=0.03, ticks=[0, 1])
cb.ax.set_yticklabels([r'void $0$', r'solid $1$'])
rho = np.linspace(0, 1, 300)
b.semilogy(rho, K + (1 - K) * rho ** 2, 'k-', lw=2.2)
b.axhline(K, color='tab:red', ls='--', lw=1.4)
b.text(0.55, 2.2 * K, r'$k = 10^{-4}$', color='tab:red')
b.set(xlabel=r'$\rho$', ylabel=r'$\mathbb{C}/\mathbb{C}_0$', xlim=(0, 1), ylim=(5e-5, 2))
b.set_box_aspect(1), b.grid(alpha=0.3)
b.set_title(r'$\mathbb{C} = [k + (1-k)\rho^2]\,\mathbb{C}_0$')

# option 2: 3 x 3 periodic tiling, unit cell outlined
s2 = sf[0, 1]
s2.suptitle(r'\textbf{2}\; periodic microstructure, one unit cell', fontsize=16)
a = s2.subplots()
small = r[::4, ::4]
cell(a, np.tile(small, (3, 3)), ext=(0, 3, 0, 3))
a.add_patch(Rectangle((1, 1), 1, 1, fill=False, ec='tab:red', lw=2.5))
a.set_xticks([]), a.set_yticks([])

# option 3: unit cell + pixel zoom
s3 = sf[1, 0]
s3.suptitle(r'\textbf{3}\; one design variable per pixel', fontsize=16)
a, b = s3.subplots(1, 2)
cell(a, r)
# a 24 x 24 pixel window centred on the interface (rho = 0.5) nearest to (0.45, 0.45)
cand = np.argwhere(np.abs(r - 0.5) < 0.05)
ic, jc = cand[np.argmin(np.hypot(cand[:, 0] / n - 0.45, cand[:, 1] / n - 0.45))]
i0, j0 = ic - 12, jc - 12
x0, y0, w = j0 / n, i0 / n, 24 / n
a.add_patch(Rectangle((x0 - 4 / n, y0 - 4 / n), w + 8 / n, w + 8 / n, fill=False, ec='tab:red', lw=2))
a.set_xticks([]), a.set_yticks([]), a.set_title(r'design $\rho(\mathbf{x})$, $2048^2$ pixels')
z = r[i0:i0 + 24, j0:j0 + 24]
cell(b, z, ext=(0, 24, 0, 24))
for t in range(25):
    b.axhline(t, color='0.6', lw=0.3), b.axvline(t, color='0.6', lw=0.3)
b.set_xticks([]), b.set_yticks([]), b.set_title(r'zoom: $24 \times 24$ pixels, $\rho_i \in [0, 1]$')
for sp in b.spines.values():
    sp.set_color('tab:red'), sp.set_linewidth(2)

# option 4: minimal schematic
s4 = sf[1, 1]
s4.suptitle(r'\textbf{4}\; schematic: solid, void, diffuse interface', fontsize=16)
a, b = s4.subplots(1, 2)
xx, yy = np.meshgrid(np.linspace(-1, 1, 400), np.linspace(-1, 1, 400))
dist = np.sqrt((xx / 0.55) ** 2 + (yy / 0.35) ** 2) - 1
sch = 1 / (1 + np.exp(-dist / 0.05))
a.imshow(sch, origin='lower', cmap='Greys', vmin=0, vmax=1, extent=(-1, 1, -1, 1))
a.text(0, 0.13, r'void $\rho = 0$, $k\,\mathbb{C}_0$', ha='center', va='center', fontsize=13)
a.text(0, -0.75, r'solid $\rho = 1$, $\mathbb{C}_0$', ha='center', va='center', color='white', fontsize=13)
a.plot([-1, 1], [0, 0], color='tab:red', lw=1.2, ls='--')
a.set_xticks([]), a.set_yticks([])
s_ = np.linspace(-1, 1, 400)
b.plot(s_, sch[200], 'k-', lw=2.2)
b.set(xlabel=r'position along the dashed line', ylabel=r'$\rho$', ylim=(-0.05, 1.05))
b.set_box_aspect(1), b.grid(alpha=0.3)
b.set_title('density profile')
out = os.path.join(HERE, 'figures', 'slide_density_options_preview.png')
fig.savefig(out, dpi=110)
print('wrote', out)

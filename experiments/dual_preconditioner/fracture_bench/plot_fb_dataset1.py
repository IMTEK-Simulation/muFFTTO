"""Fracture-Bench Dataset 1 (periodic analogue, fb_dataset1_AT2.py): damage at the last step (exact pixels, hole and
notches white), macro stress vs strain, CG per elastic solve per load step, Green vs dual (x2 = cost).
    python plot_fb_dataset1.py  ->  figures/fb_dataset1_results.png"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
TAG = '_fb_hole_holes0.001_l0.0068182_k0.001_biax'
R = {}
for P in ('Green', 'dual'):
    with netCDF4.Dataset(os.path.join(HERE, 'exp_data', f'AT2_N128_{P}{TAG}.nc')) as ds:
        R[P] = {k: np.asarray(ds[k][:]) for k in ('damage', 'load', 'macro_stress', 'staggered_iteration', 'cg_u_step')}
N = 128
h = 1.0 / N
c = np.stack(np.meshgrid(np.arange(N) * h, np.arange(N) * h, indexing='ij'))
x, y = c[0] - 0.5, c[1] - 0.5
hole = x ** 2 + y ** 2 < (0.5 / 2.2) ** 2
for k in range(8):
    th = k * np.pi / 4
    s, t = x * np.cos(th) + y * np.sin(th), -x * np.sin(th) + y * np.cos(th)
    hole |= (s >= 0.5 / 2.2 - 2 * h) & (s <= 0.5 / 2.2 + 0.05 / 2.2) & (np.abs(t) < 0.75 * h)

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), layout='constrained')
d = R['dual']['damage'][-1, 0]
img = plt.get_cmap('inferno_r')(np.clip(d, 0, 1))
img[hole] = (1, 1, 1, 1)
ax[0].imshow(np.transpose(img, (1, 0, 2)), origin='lower', interpolation='nearest', extent=(0, 2.2, 0, 2.2))
ax[0].set_title(f'damage, last step ($\\bar\\varepsilon$ = {R["dual"]["load"][-1]:.4f}), max d = {d.max():.2f}')
fig.colorbar(plt.cm.ScalarMappable(cmap='inferno_r', norm=plt.Normalize(0, 1)), ax=ax[0], label='d')
for P, col, f in (('Green', 'tab:green', 1), ('dual', 'tab:red', 2)):
    r = R[P]
    ax[1].plot(r['load'] * 1e3, r['macro_stress'][:, 0, 0] * 1e3, '-o' if P == 'Green' else '--', color=col, ms=3, label=P)
    ax[2].plot(np.arange(1, len(r['load']) + 1), f * r['cg_u_step'] / r['staggered_iteration'], '-o', color=col, ms=3,
               label=f'{P}' + (r' ($\times 2$)' if f == 2 else '') + f', total {f * r["cg_u_step"].sum():.0f}')
ax[1].set(xlabel=r'biaxial strain $\bar\varepsilon$ [$10^{-3}$]', ylabel=r'$\bar\sigma_{xx}/\mu$ [$10^{-3}$]',
          title='macro stress (curves coincide)')
ax[2].set(xlabel='load step', ylabel='CG per elastic solve', title='elastic CG (dual counted x2)', ylim=(0, None))
for a in ax[1:]:
    a.grid(alpha=0.3), a.legend()
fig.suptitle('Fracture-Bench Dataset 1, periodic analogue: hole + 8 notches, equal biaxial strain, 128$^2$, '
             r'$\ell/h = 0.87$, $\eta = 10^{-3}$, their 44 steps')
out = os.path.join(HERE, 'figures', 'fb_dataset1_results.png')
fig.savefig(out, dpi=130)
print(out)

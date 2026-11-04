"""
Crack path vs preferred preconditioner, 256^2 (2D). One row per case: the final damage (exact pixels; disks white),
the macroscopic stress and the elastic CG iterations per solve (mean over each load step, raw iterations: the dual's
cost is 2 per iteration) of Green / Green-Jacobi / dual over the load. The title gives the work of the cracking phase
(after the peak, incl. the breaking step; dual 2 per iteration) and the ratio best other / dual (> 1: dual cheaper).

usage:  python plot_AT2_crack_gallery.py       writes figures/AT2_crack_gallery.png
"""
import os
import sys

import numpy as np
import netCDF4
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '../../..')))
from muFFTTO import microstructure_library  # noqa: E402

N = 256
PRECS = [('Green', '#1baf7a', '--', 1), ('Green_Jacobi', '#2a78d6', ':', 1), ('dual', '#d6402a', '-', 2)]
LABELS = {'Green': 'Green', 'Green_Jacobi': 'Green-Jacobi', 'dual': 'dual'}
CASES = [('pores 1e-3 + damage in pores, l/h = 4: stepwise crack', '_l0.015625_lh'),
         ('pores 1e-3 + damage in pores, l = 1/16 (l/h = 16)', '_l0.0625_smooth'),
         ('pores = k = 1e-3 (no damage in pores), l = 1/16', '_holes0.001_l0.0625_k0.001_smooth'),
         ('pores = k = 1e-4 (no damage in pores), l/h = 2', '_holes0.0001_l0.0078125_smooth'),
         ('stiff inclusions 10x, l/h = 2', '_incl10_l0.0078125_smooth'),
         ('stiff inclusions 10x, l = 1/16', '_incl10_l0.0625_smooth'),
         ('stiff inclusions 1000x, l/h = 2', '_incl1000_l0.0078125_smooth'),
         ('stiff inclusions 1000x, l = 1/16', '_incl1000_l0.0625_smooth')]
x = np.arange(N) / N
disk = microstructure_library.get_geometry(nb_voxels=(N, N), microstructure_name='polydisperse_pores',
                                           coordinates=np.array(np.meshgrid(x, x, indexing='ij'))) == 0

fig, axes = plt.subplots(len(CASES), 2, figsize=(15, 3.6 * len(CASES)), layout='constrained', dpi=110,
                         gridspec_kw=dict(width_ratios=[1, 2.6]))
for row, (title, tag) in enumerate(CASES):
    base = os.path.join(HERE, 'exp_data', f'AT2_N{N}_{{}}_polydisperse_pores{tag}')
    with netCDF4.Dataset(base.format('dual') + '.nc') as ds:
        nodal = np.asarray(ds['damage'][-1, 0])
        steps, sxx, load = (np.asarray(ds['step'][:]).astype(int), np.asarray(ds['macro_stress'][:, 0, 0]),
                            np.asarray(ds['load'][:]))
    pix = 0.25 * (nodal + np.roll(nodal, -1, 0) + np.roll(nodal, -1, 1) + np.roll(np.roll(nodal, -1, 0), -1, 1))
    img = plt.cm.inferno_r(np.clip(pix, 0, 1) * 0.9)
    img[disk] = (1, 1, 1, 1)
    ax_d, ax_c = axes[row]
    ax_d.imshow(np.transpose(img, (1, 0, 2)), origin='lower', interpolation='nearest', extent=(0, 1, 0, 1))
    ax_d.set_xticks([]), ax_d.set_yticks([])
    peak = steps[sxx.argmax()]
    broken = steps[(steps > peak) & (sxx < 0.2 * sxx.max())]
    first_broken = broken.min() if len(broken) else steps.max()
    work = {}
    for p, c, ls, fac in PRECS:
        f = base.format(p) + '_iterations.csv'
        if not os.path.exists(f):
            continue
        d = np.genfromtxt(f, delimiter=',', names=True, dtype=None, encoding=None)
        st = np.unique(d['step'])
        ax_c.plot([d['load'][d['step'] == s][0] for s in st], [d['cg_u'][d['step'] == s].mean() for s in st],
                  color=c, ls=ls, lw=2, label=LABELS[p])
        m = (d['step'] > peak) & (d['step'] <= first_broken)
        work[p] = fac * d['cg_u'][m].mean()
    tw = ax_c.twinx()
    tw.plot(load, sxx, color='0.6', lw=1.2)
    tw.set_ylim(0, 1.15 * sxx.max()), tw.set_ylabel(r'$\bar\sigma_{xx}$ (grey)', fontsize=10, color='0.45')
    tw.tick_params(labelsize=9, colors='0.45')
    ax_c.set_ylim(0, None), ax_c.set_xlim(0, load[-1])
    ax_c.set_ylabel('CG iterations per solve', fontsize=11), ax_c.tick_params(labelsize=10)
    ax_c.grid(True, color='0.92')
    if row == len(CASES) - 1:
        ax_c.set_xlabel(r'macroscopic strain $\bar\varepsilon_{xx}$', fontsize=11)
    ratio = min(work.get('Green', np.inf), work.get('Green_Jacobi', np.inf)) / work['dual']
    ax_c.set_title(f'{title}\ncracking work G / GJ / dual(2x) = '
                   + ' / '.join(f'{work.get(p, np.nan):.0f}' for p, *_ in PRECS)
                   + f'   ->   best other / dual = {ratio:.2f}' + ('  (dual wins)' if ratio > 1 else '  (dual loses)'),
                   fontsize=11.5, loc='left')
    if row == 0:
        ax_c.legend(fontsize=10, loc='upper left')
out = os.path.join(HERE, 'figures', 'AT2_crack_gallery.png')
fig.savefig(out)
print(f'wrote {out}')

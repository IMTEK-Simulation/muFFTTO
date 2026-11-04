"""
64^2 small test (run_AT2_incl_lh_64.sh): pores vs stiff tough inclusions (10x), l/h = 2 and 4.
Top: final damage of the dual run, exact pixels. Middle: stress-strain curve.
Bottom: elastic CG iterations per solve along the load path (cost: dual counted x2, dashed).

usage:  python plot_AT2_incl_lh_64.py       writes figures/AT2_N64_incl_lh.png
"""
import os

import numpy as np
import netCDF4
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
CASES = [('_l0.03125', 'pores, l/h = 2'), ('_l0.0625', 'pores, l/h = 4'),
         ('_incl10_l0.03125', 'inclusions 10x, l/h = 2'), ('_incl10_l0.0625', 'inclusions 10x, l/h = 4')]
PRECONDS = [('Green', 'C2', 'Green'), ('Green_Jacobi', 'C0', 'Green-Jacobi'), ('dual', 'C3', 'dual')]


def iterations(p, tag):
    return np.genfromtxt(os.path.join(HERE, 'exp_data', f'AT2_N64_{p}_polydisperse_pores{tag}_iterations.csv'),
                         delimiter=',', names=True)


fig, axes = plt.subplots(3, 4, figsize=(17, 12), layout='constrained', dpi=150)
for col, (tag, title) in enumerate(CASES):
    with netCDF4.Dataset(os.path.join(HERE, 'exp_data', f'AT2_N64_dual_polydisperse_pores{tag}.nc')) as ds:
        d = np.asarray(ds['damage'][-1, 0])
        load = np.asarray(ds['load'][:])
        sxx = np.asarray(ds['macro_stress'][:, 0, 0])
    ax = axes[0, col]
    ax.imshow(d.T, origin='lower', vmin=0, vmax=1, cmap='inferno', interpolation='nearest', extent=(0, 1, 0, 1))
    ax.set_title(f'{title}\nfinal damage (dual), E_xx = {load[-1]:.3f}', fontsize=10)
    ax.set_xticks([]), ax.set_yticks([])
    axes[1, col].plot(np.r_[0, load], np.r_[0, sxx], 'o-', ms=2.5, color='k')
    axes[1, col].set_xlabel(r'$\bar\varepsilon_{xx}$'), axes[1, col].set_ylabel(r'$\bar\sigma_{xx}$')
    axes[1, col].grid(True, color='0.9')
    ax = axes[2, col]
    for p, c, label in PRECONDS:
        it = iterations(p, tag)
        k = np.arange(len(it))
        ax.plot(k, it['cg_u'], '-', lw=0.8, color=c,
                label=f'{label}: mean {it["cg_u"].mean():.0f}' + (f' (cost {2 * it["cg_u"].mean():.0f})' if p == 'dual' else ''))
        if p == 'dual':
            ax.plot(k, 2 * it['cg_u'], '--', lw=0.8, color=c, alpha=0.6, label='dual x2 (cost)')
    ax.set_xlabel('elastic solve (all staggered iterations)'), ax.set_ylabel('CG iterations')
    ax.set_yscale('log'), ax.grid(True, color='0.9', which='both')
    ax.legend(fontsize=8)
out = os.path.join(HERE, 'figures', 'AT2_N64_incl_lh.png')
fig.savefig(out)
print(f'wrote {out}')

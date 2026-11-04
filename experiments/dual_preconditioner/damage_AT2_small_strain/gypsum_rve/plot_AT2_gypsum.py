"""
Gypsum RVE (see README.md): convergence of the physics and cost of the preconditioners over the grid at the physical
length scale l = 0.5 mm (l/h = 1.25 / 2.5 / 5 / 10 on 64^2 / 128^2 / 256^2 / 512^2). One row per microstructure
(porous gypsum, stiff grains):

    left    macroscopic stress-strain curves of the dual runs on every grid (MPa: unit E = 600 MPa)
    middle  peak stress against l/h (all preconditioners give the same curve)
    right   total elastic work of the run (CG iterations x K applications, dual 2) against the grid

usage:  python plot_AT2_gypsum.py      writes figures/AT2_gypsum.png and prints the table
"""
import os

import numpy as np
import netCDF4
import matplotlib
import matplotlib.ticker

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
GRIDS = (64, 128, 256, 512, 1024)
L_MM, H_MM = 0.5, 25.6
PRECS = [('Green', 'C2', 'o', 1, 'Green'), ('Green_Jacobi', 'C0', 's', 1, 'Green-Jacobi'), ('dual', 'C3', '^', 2, 'dual')]
VARIANTS = [('_holes0.0001', 'porous gypsum (pores of stiffness k, no damage in them)'),
            ('_incl10', 'gypsum with stiff tough grains (10x)')]


def load(n, p, tag):
    base = os.path.join(HERE, 'exp_data', f'AT2_N{n}_{p}_polydisperse_pores{tag}_l0.0195915_gypsum')
    log = base.replace('exp_data', 'logs') + '.log'
    if not (os.path.exists(log) and 'Elapsed time' in open(log).read()):
        return None
    with netCDF4.Dataset(base + '.nc') as ds:
        strain, stress = np.asarray(ds['load'][:]), np.asarray(ds['macro_stress'][:, 0, 0]) * 600.0
    it = np.genfromtxt(base + '_iterations.csv', delimiter=',', names=True, dtype=None, encoding=None)
    return strain, stress, it


fig, axes = plt.subplots(2, 3, figsize=(19, 11), layout='constrained', dpi=140)
print(f'{"variant":12s} {"N":>4s} {"l/h":>5s} | ' + ' '.join(f'{q[4]:>13s}' for q in PRECS) + ' | peak MPa | solves')
for row, (tag, title) in enumerate(VARIANTS):
    ax_s, ax_p, ax_w = axes[row]
    shades = plt.cm.Reds(np.linspace(0.35, 0.95, len(GRIDS)))
    peaks = {}
    for n, sh in zip(GRIDS, shades):
        lh = L_MM / (H_MM / n)
        r = load(n, 'dual', tag)
        if r is not None:
            ax_s.plot(np.r_[0, r[0]] * 1e3, np.r_[0, r[1]], color=sh, lw=2, label=f'{n}$^2$ (l/h = {lh:g})')
        works = []
        for p, c, m, fac, label in PRECS:
            q = load(n, p, tag)
            works.append(None if q is None else fac * q[2]['cg_u'].sum())
            if q is not None:
                peaks.setdefault(n, q[1].max())
        if any(w is not None for w in works):
            q = next(load(n, p, tag) for p, *_ in PRECS if load(n, p, tag) is not None)
            print(f'{"pores" if row == 0 else "grains":12s} {n:4d} {lh:5.2f} | '
                  + ' '.join(f'{w:13.0f}' if w is not None else '            -' for w in works)
                  + f' | {peaks[n]:8.3f} | {len(q[2]):6d}')
    ax_s.set_xlabel(r'$\bar\varepsilon_{xx}$ [10$^{-3}$]', fontsize=12), ax_s.set_ylabel(r'$\bar\sigma_{xx}$ [MPa]', fontsize=12)
    ax_s.set_title(f'{title}\nstress-strain (dual runs)', fontsize=12.5), ax_s.legend(fontsize=10)
    lhs = [L_MM / (H_MM / n) for n in peaks]
    ax_p.plot(lhs, list(peaks.values()), 'ko-', ms=8, lw=2)
    ax_p.axvspan(5, 24, color='0.92', zorder=0)
    ax_p.text(5.3, min(peaks.values()), 'l/h >= 5:\nmesh objective', fontsize=10, color='0.35', va='bottom')
    ax_p.set_xscale('log', base=2), ax_p.set_xticks([1.25, 2.5, 5, 10, 20]), ax_p.set_xticklabels(["1.25", "2.5", "5", "10", "20"])
    ax_p.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax_p.set_xlabel('l / h  (l = 0.5 mm fixed; grid 64$^2$ ... 1024$^2$)', fontsize=12)
    ax_p.set_ylabel('peak stress [MPa]', fontsize=12), ax_p.set_title('convergence of the peak stress', fontsize=12.5)
    for p, c, m, fac, label in PRECS:
        pts = [(n, fac * load(n, p, tag)[2]['cg_u'].sum() / 1e3) for n in GRIDS if load(n, p, tag) is not None]
        if pts:
            ax_w.plot(*np.array(pts).T, '-', marker=m, ms=8, lw=2, color=c, label=label)
            for x, y in pts:
                ax_w.annotate(f'{y:.0f}', (x, y), textcoords='offset points', xytext=(6, 4), fontsize=10, color=c)
    ax_w.set_xscale('log', base=2), ax_w.set_xticks(GRIDS), ax_w.set_xticklabels([f'{n}$^2$' for n in GRIDS])
    ax_w.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax_w.set_ylim(0, None)
    ax_w.set_xlabel('grid', fontsize=12), ax_w.set_ylabel('total elastic work [thousands]\n(dual 2 per iteration)', fontsize=12)
    ax_w.set_title('cost of the whole run', fontsize=12.5), ax_w.legend(fontsize=11)
    for ax in axes[row]:
        ax.tick_params(labelsize=11), ax.grid(True, color='0.9')
out = os.path.join(HERE, 'figures', 'AT2_gypsum.png')
fig.savefig(out)
print(f'wrote {out}')

"""Figures of the small-strain AT2 runs (example_2D_phase_field_fracture_AT2.py, exp_data/AT2_N<n>_<prec>.nc, logs/):

    figures/AT2_crack_N<n>_dual.gif   movie of the damage field over the load steps (for viewing, not for the paper)
    <paper>/figures/part2_AT2.pdf     damage at three load levels (256^2), stress-strain curves (64^2-256^2) and the
                                      elastic CG iterations per load step (256^2), drawn at the text width 4.8 in
and the LaTeX table rows (total CG iterations) on stdout.

    python plot_AT2.py [N_MOVIE]
"""
import os
import re
import sys

import matplotlib
import matplotlib.ticker
import netCDF4
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
PAPER = os.path.expanduser('~/PycharmProjects/Paper-Double-Green-Preconditioner/figures')
GRIDS = [64, 128, 256]
PRECS = [('Green', 'Green', '#1baf7a'), ('Green_Jacobi', 'Green-Jacobi', '#2a78d6'), ('dual', 'dual', '#d6402a')]
N_MOVIE = int(sys.argv[1]) if len(sys.argv) > 1 else 256
TAG = sys.argv[2] if len(sys.argv) > 2 else ''          # e.g. _l0.03125
if TAG:
    GRIDS = [32, 64, 128, 256, 512]


def frames(N, p):
    f = os.path.join(HERE, 'exp_data', f'AT2_N{N}_{p}{TAG}.nc')
    if not os.path.exists(f):
        return None
    d = netCDF4.Dataset(f)
    return dict(damage=np.asarray(d['damage'][:, 0]), load=np.asarray(d['load'][:]),
                stress=np.asarray(d['macro_stress'][:, 0, 0]))


def cg_per_step(N, p):
    log = os.path.join(HERE, 'logs', f'AT2_N{N}_{p}{TAG}.log')
    out, total = [], None
    for line in open(log):
        m = re.search(r'CG its \(u, d\) = \((\d+), (\d+)\)', line)
        if m:
            out.append(int(m[1]))
        m = re.search(r'total CG iterations elasticity = (\d+), damage = (\d+)', line)
        if m:
            total = (int(m[1]), int(m[2]))
    return np.array(out), total


# paper figure -------------------------------------------------------------------------------------------------------
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'legend.fontsize': 6})
fig = plt.figure(figsize=(4.8, 3.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
r = frames(N_MOVIE, 'dual')
d_max = r['damage'].max(axis=(1, 2))
picks = [int(np.argmax(r['stress'])) // 2, int(np.argmax(r['stress'])), len(r['load']) - 1]   # before, at, after the peak
for col, k in enumerate(picks):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(r['damage'][k].T, origin='lower', cmap='magma', vmin=0, vmax=1, interpolation='nearest',
              rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'$\\bar\\varepsilon_{{xx}}={r["load"][k]:.3f}$')
ax = fig.add_subplot(gs[1, 0:2])
for N, ls in zip(GRIDS, [(0, (1, 1)), '-.', ':', '--', '-']):
    rr = frames(N, 'dual')
    if rr is None or cg_per_step(N, 'dual')[1] is None:      # not finished (yet)
        continue
    ax.plot(np.r_[0, rr['load']], np.r_[0, rr['stress']], color='0.2', ls=ls, lw=1.2, label=f'${N}^2$')
ax.set_xlabel(r'macroscopic strain $\bar\varepsilon_{xx}$')
ax.set_ylabel(r'$\bar\sigma_{xx}$')
ax.legend(frameon=False)
ax.grid(alpha=0.3)
ax2 = fig.add_subplot(gs[1, 2])
for p, label, c in PRECS:
    cg, _ = cg_per_step(N_MOVIE, p)
    if len(cg):
        ax2.semilogy(np.arange(1, len(cg) + 1), cg, color=c, lw=1.0, label=label)
ax2.set_xlabel('load step')
ax2.set_ylabel(f'elastic CG its, ${N_MOVIE}^2$')
ax2.legend(frameon=False, loc='upper left', handlelength=1.2)
ax2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([30, 50, 100, 200, 400]))
ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax2.grid(alpha=0.3, which='both')
for a in (ax, ax2):
    for side in ('top', 'right'):
        a.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(PAPER, 'part2_AT2' + TAG.replace('.', 'p') + '.pdf'), dpi=300)
plt.close(fig)

print('% grid & Green & Green-Jacobi & dual (elastic CG total) & damage CG & final sigma_xx')
for N in GRIDS:
    tot = {p: cg_per_step(N, p)[1] for p, *_ in PRECS}
    if tot['dual'] is None:
        continue
    sig = frames(N, 'dual')['stress'][-1]
    print(f'${N}^2$ & ' + ' & '.join('--' if tot[p] is None else f'{tot[p][0]:,}'.replace(',', '\\,')
                                     for p, *_ in PRECS)
          + f' & {tot["dual"][1]:,}'.replace(',', '\\,') + f' & ${sig:.3e}$ \\\\'.replace('e-0', r'\cdot10^{-') .replace(r'$ \\', r'}$ \\'))

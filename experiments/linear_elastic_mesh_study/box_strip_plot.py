"""
Figure of run_box_strip.sh: inner PCG on S = P G0 P preconditioned by the local Schur complement on a strip of width w
around the fixed lines, K_FF - K_FIw K_IwIw^-1 K_IwF, balanced on the translations (2D square, all four faces clamped).

    (a) cond(C^-1 S) vs N at fixed strip width in pixels     (b) the same at fixed physical width w / N
    (c) inner steps per outer iteration vs N

usage:  python box_strip_plot.py      -> exp_data/box_strip.png (copied to report/figs)
"""
import os
import shutil

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.realpath(__file__))
RUNS = os.path.join(HERE, 'exp_data', 'runs_box')
NS = (32, 64, 128)
WS = (1, 2, 4, 8)
plt.rcParams.update({'font.size': 12})
STRIP = {w: plt.get_cmap('Purples')(0.4 + 0.55 * i / 3) for i, w in enumerate(WS)}
KFF, FAMILY = 'tab:red', 'tab:orange'


def load(n, tag):
    f = os.path.join(RUNS, f'N{n}_box_{tag}.npz')
    return {k: v[()] if v.ndim == 0 else v for k, v in np.load(f).items()} if os.path.exists(f) else None


def cond(n, key):
    r = load(n, 'dense')
    return r[f'eig_{key}'][-1] / r[f'eig_{key}'][0] if r is not None and f'eig_{key}' in r else None


def line(ax, pts, colour, text, ls='-', dy=-4):
    pts = [(x, y) for x, y in pts if y is not None]
    if pts:
        ax.plot(*zip(*pts), ls=ls, marker='o', color=colour, lw=2, ms=6)
        ax.annotate(f'{text}: {pts[-1][1]:.1f}', pts[-1], textcoords='offset points', xytext=(8, dy), fontsize=10.5,
                    color=colour)


fig, axes = plt.subplots(1, 3, figsize=(20, 6), layout='constrained')
ax = axes[0]
line(ax, [(n, cond(n, 'kff')) for n in NS], KFF, '$\\mathbf{K}_{FF}$ (w = 0)', ls=':')
line(ax, [(n, cond(n, 'family')) for n in NS], FAMILY, 'family FFT', ls='--')
for w in WS:
    line(ax, [(n, cond(n, f'strip{w}bal')) for n in NS], STRIP[w], f'strip w = {w} px')
ax.set(yscale='log', ylabel='cond($\\mathbf{C}^{-1}\\mathbf{S}$)',
       title='(a) fixed strip width in pixels: still grows like 1/h')
ax = axes[1]
line(ax, [(n, cond(n, 'family')) for n in NS], FAMILY, 'family FFT', ls='--')
for frac, colour in ((32, STRIP[1]), (16, STRIP[4]), (8, STRIP[8])):
    line(ax, [(n, cond(n, f'strip{n // frac}bal')) for n in NS if n // frac in WS], colour, f'w = N/{frac}')
ax.set(yscale='log', ylim=(1, 200), ylabel='cond($\\mathbf{C}^{-1}\\mathbf{S}$)',
       title='(b) fixed physical width w/N: nearly bounded')
ax = axes[2]
for tag, colour, text, ls in (('kff1e-06', KFF, '$\\mathbf{K}_{FF}$', ':'), ('family1e-06', FAMILY, 'family FFT', '--')):
    line(ax, [(n, r['inner_its'].mean()) for n in NS if (r := load(n, tag)) is not None], colour, text, ls=ls)
for w in WS:
    line(ax, [(n, r['inner_its'].mean()) for n in NS if (r := load(n, f'strip{w}1e-06')) is not None], STRIP[w],
         f'strip w = {w} px', dy=-10 if w == 8 else -4)
ax.set(ylim=(0, None), ylabel='inner steps per outer iteration (tolerance 1e-6)', title='(c) inner PCG steps')
for ax in axes:
    ax.set(xscale='log', xlabel='N (pixels over the square)')
    ax.set_xticks(NS, [str(n) for n in NS])
    ax.set_xlim(26, 400)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(True, color='0.92', which='both')
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Inner solve of $\\mathbf{S}$ (2D square, all four faces clamped) preconditioned by the local Schur complement '
             '$\\mathbf{K}_{FF} - \\mathbf{K}_{FI_w}\\mathbf{K}_{I_wI_w}^{-1}\\mathbf{K}_{I_wF}$ on a strip of width w '
             'around the fixed lines', fontsize=13)
out = os.path.join(HERE, 'exp_data', 'box_strip.png')
fig.savefig(out, dpi=110)
shutil.copy(out, os.path.join(HERE, 'report', 'figs', 'box_strip.png'))
print('wrote', out)

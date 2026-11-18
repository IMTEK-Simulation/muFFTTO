"""
Plots of the 2D noise-start test (noise_start_2D.py, run_noise_start_2D.sh), all with the L-BFGS first-step fix 0.05:
  white noise drawn on the grid itself (64^2, 128^2, 256^2)  vs  the same 64^2 noise field interpolated to
  256^2 / 512^2 / 1024^2 (grid-independent start). Whatever exists in logs/noise_start_2D is plotted.
    python plot_noise_start_2D.py  ->  figures/noise_start_2D_designs.png (designs on their exact pixels),
                                       figures/noise_start_2D_curves.png
"""
import glob
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, 'logs', 'noise_start_2D')
FIRST = os.environ.get('FIRST_STEP', '0.05')
runs = []
for path in glob.glob(os.path.join(LOGS, f'N*_noise*_first{FIRST}_seed0.npz')):
    n, m = map(int, re.search(r'N(\d+)_noise(\d+)_', path).groups())
    runs.append((m == n, n, m, np.load(path)))
runs.sort(key=lambda r: (not r[0], r[1]))                  # white noise first, then coarse noise; by grid
plt.rcParams.update({'font.size': 13})


def label(white, n, m):
    return f'{n}², white noise' if white else f'{n}², {m}² noise interpolated'


# ---- designs: one row per run (start, iteration 20, final)
fig, axes = plt.subplots(len(runs), 3, figsize=(12.5, 4.3 * len(runs)), squeeze=False)
for row, (white, n, m, data) in enumerate(runs):
    its = list(data['snapshot_iterations'])
    shown = [(0, data['snapshots'][0], 'start')]
    if 20 in its and 20 < len(data['f']):
        shown.append((20, data['snapshots'][its.index(20)], 'iteration 20'))
    shown.append((len(data['f']), data['final'], 'final'))
    for col in range(3):
        ax = axes[row, col]
        ax.set_xticks([]), ax.set_yticks([])
        if col >= len(shown):
            ax.set_axis_off()
            continue
        it, image, what = shown[col]
        ax.imshow(image, cmap='gray_r', vmin=0, vmax=1, interpolation='nearest', origin='lower')
        info = '' if it == 0 else f': it. {it}, f {data["f"][it - 1]:.3g}, nu {data["nu"][it - 1]:+.2f}'
        ax.set_title(f'{label(white, n, m)}\n{what}{info}', fontsize=12)
fig.suptitle('2D TO from random noise (eta = 2h, L-BFGS first-step fix 0.05)', fontsize=15)
fig.tight_layout()
out = os.path.join(HERE, 'figures', 'noise_start_2D_designs.png')
fig.savefig(out, dpi=max([r[1] for r in runs] + [400]) // 4)          # >= 1 screen pixel per design pixel
print('saved', out)

# ---- curves: hue = kind of start (white: red, interpolated 64^2 noise: blue), shade = grid
fig, (ax_f, ax_s, ax_n) = plt.subplots(1, 3, figsize=(19, 5.8))
grids = sorted({r[1] for r in runs})
for white, n, m, data in runs:
    shade = 0.45 + 0.5 * grids.index(n) / max(len(grids) - 1, 1)
    colour = plt.get_cmap('Reds' if white else 'Blues')(shade)
    it = np.arange(1, len(data['f']) + 1)
    kw = dict(color=colour, linestyle='-' if white else '--', linewidth=2.2, marker='o' if white else 's',
              markevery=max(len(it) // 8, 1), markersize=6, label=label(white, n, m))
    ax_f.plot(data['evaluations'], data['f'], **kw)
    ax_s.plot(it, data['std'], **kw)
    ax_n.plot(it, data['nu'], **kw)
ax_f.set_yscale('log')
ax_f.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
ax_f.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax_f.set(xlabel='objective evaluations', ylabel='objective f', xlim=(0, None), title='objective')
ax_s.set(xlabel='L-BFGS iteration', ylabel='std of the design (0 = uniform)', xlim=(0, None), ylim=(0, 0.55),
         title='design contrast')
ax_n.axhline(-0.5, color='0.5', linewidth=1, linestyle=':')
ax_n.set(xlabel='L-BFGS iteration', ylabel='Poisson ratio nu12', xlim=(0, None), title='nu12 (target -0.5)')
for ax in (ax_f, ax_s, ax_n):
    ax.grid(alpha=0.3)
handles, labels = ax_f.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=12)
fig.tight_layout(rect=(0, 0.14, 1, 1))
out = os.path.join(HERE, 'figures', 'noise_start_2D_curves.png')
fig.savefig(out, dpi=110)
print('saved', out)

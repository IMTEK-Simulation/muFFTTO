"""
Figure of the 2D mesh-independence check (run_mesh_independence_2D.sh): for the three series
    2h     eta = 2 h    (interface shrinks with the grid)
    eta16  eta = 1/16   (fixed physical interface width)
    eta32  eta = 1/32   (fixed physical interface width)
top row     final interface penalty, stiffness misfit and Poisson ratio against the grid size
rows below  the final designs (solid dark), one row per series, one column per grid, on their exact pixels.
    python plot_mesh_independence_2D.py     -> figures/mesh_independence_2D.png  (+ table on stdout)
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SERIES = {'2h': ('interface width 2 h (shrinks with the grid)', '#2a78d6', '-', 'o'),
          'eta16': ('interface width 1/16 (fixed)', '#eb6834', '--', 's'),
          'eta32': ('interface width 1/32 (fixed)', '#1baf7a', '-.', '^')}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 13, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False})

runs = {}
for path in sorted((HERE / 'logs' / 'mesh_independence').glob('N*_nu*_s*_*.npz')):
    data = np.load(path)
    runs[(path.stem.split('_')[-1], int(path.stem.split('_')[0][1:]))] = data
if ('2h', 32) in runs:                                   # 2 h at 32^2 is eta = 1/16, 2 h at 64^2 is eta = 1/32
    runs[('eta16', 32)] = runs[('2h', 32)]
if ('2h', 64) in runs:
    runs[('eta32', 64)] = runs[('2h', 64)]
grids = sorted({n for _, n in runs})

print(f'{"series":<7s} {"N":>4s} {"eta / h":>8s} {"f":>8s} {"misfit":>8s} {"interface":>10s} {"nu12":>8s} '
      + ' '.join(f'{name:>9s}' for name in ('C1111', 'C1122', 'C2222', 'C1212')))
for (series, n), data in sorted(runs.items()):
    print(f'{series:<7s} {n:4d} {float(data["eta"]) * n:8.1f} {data["f"][-1]:8.4f} {data["misfit"][-1]:8.4f} '
          f'{data["interface"][-1]:10.4f} {data["nu"][-1]:+8.4f} ' + ' '.join(f'{w:+9.5f}' for w in data['energies'][-1]))

fig = plt.figure(figsize=(4.3 * len(grids) + 1, 5.0 + 4.3 * len(SERIES)))
outer = fig.add_gridspec(2, 1, height_ratios=[4.2, 4.3 * len(SERIES)], hspace=0.12)
top = outer[0].subgridspec(1, 3, wspace=0.3)
quantities = [('interface', 'interface penalty (final)'), ('misfit', 'stiffness misfit (final)'),
              ('nu', 'Poisson ratio  C1122 / C2222 (final)')]
for column, (key, title) in enumerate(quantities):
    ax = fig.add_subplot(top[column])
    for series, (label, colour, style, marker) in SERIES.items():
        ns = [n for n in grids if (series, n) in runs]
        ax.plot(range(len(ns)) if False else [grids.index(n) for n in ns], [float(runs[(series, n)][key][-1]) for n in ns],
                color=colour, linestyle=style, marker=marker, markersize=9, linewidth=2, label=label)
    ax.set_xticks(range(len(grids)), [f'{n}²' for n in grids])
    ax.set_xlabel('grid')
    ax.set_title(title, loc='left', fontsize=13)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if key != 'nu':
        ax.set_ylim(0, None)
    if column == 0:
        ax.legend(frameon=False, fontsize=11, loc='lower left')

bottom = outer[1].subgridspec(len(SERIES), len(grids), wspace=0.06, hspace=0.16)
for row, (series, (label, colour, _, _)) in enumerate(SERIES.items()):
    for column, n in enumerate(grids):
        ax = fig.add_subplot(bottom[row, column])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(colour)
            spine.set_linewidth(2)
        if (series, n) not in runs:
            ax.text(0.5, 0.5, 'not run', ha='center', va='center', color=MUTED, transform=ax.transAxes)
        else:
            data = runs[(series, n)]
            ax.imshow(1 - data['final'].T, cmap='gray', vmin=0, vmax=1, origin='lower', interpolation='nearest')
            ax.set_title(f'{n}²,  interface width {float(data["eta"]) * n:g} h', fontsize=12, loc='left')
        if column == 0:
            ax.set_ylabel(label.split(' (')[0], fontsize=12)
fig.suptitle('2D check: is the result mesh-size independent?  (solid = dark; same start, 600 iterations)',
             x=0.02, ha='left', fontsize=15, y=0.995)
out = HERE / 'figures'
out.mkdir(exist_ok=True)
fig.savefig(out / 'mesh_independence_2D.png', dpi=140, bbox_inches='tight')
print(out / 'mesh_independence_2D.png')

# ---- second figure: interface details. Per run the first interface on the line y = 1/2 is located (rho crosses 1/2);
# shown are a window of physical width 0.2 around it (exact pixels) and, last column, rho along that line against the
# physical distance from the crossing (one marker per node), all grids of the series in one panel.
WINDOW = 0.2
SHADES = ['#9ec5f4', '#5598e7', '#256abf', '#0d366b']                    # light -> dark = coarse -> fine grid
MARKERS = ['o', 's', '^', 'D']
fig = plt.figure(figsize=(3.6 * len(grids) + 6.5, 4.0 * len(SERIES)))
layout = fig.add_gridspec(len(SERIES), len(grids) + 1, width_ratios=[1] * len(grids) + [1.8], wspace=0.12, hspace=0.3)
for row, (series, (label, colour, _, _)) in enumerate(SERIES.items()):
    profile_ax = fig.add_subplot(layout[row, -1])
    for column, n in enumerate(grids):
        ax = fig.add_subplot(layout[row, column])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(colour)
            spine.set_linewidth(2)
        if (series, n) not in runs:
            ax.text(0.5, 0.5, 'not run', ha='center', va='center', color=MUTED, transform=ax.transAxes)
            continue
        rho = runs[(series, n)]['final']
        j = n // 2
        line = rho[:, j]
        crossings = np.nonzero((line - 0.5) * (np.roll(line, -1) - 0.5) < 0)[0]
        i = int(crossings[0]) if len(crossings) else n // 2
        half = max(2, int(round(WINDOW * n / 2)))
        index = np.arange(i - half, i + half + 1)
        window = rho[np.ix_(index % n, np.arange(j - half, j + half + 1) % n)]
        ax.imshow(1 - window.T, cmap='gray', vmin=0, vmax=1, origin='lower', interpolation='nearest')
        ax.axhline(half, color=colour, linewidth=1)
        ax.set_title(f'{n}²,  interface width {float(runs[(series, n)]["eta"]) * n:g} h', fontsize=12, loc='left')
        if column == 0:
            ax.set_ylabel(label.split(' (')[0], fontsize=12)
        fraction = (0.5 - line[i]) / (line[(i + 1) % n] - line[i]) if len(crossings) else 0.0
        sign = 1.0 if line[(i + 1) % n] > line[i] else -1.0                     # always void (left) -> solid (right)
        distance = sign * (index - i - fraction) / n
        order = np.argsort(distance)
        profile_ax.plot(distance[order], line[index % n][order], color=SHADES[column % 4], marker=MARKERS[column % 4],
                        markersize=6, linewidth=1.8, label=f'{n}²')
    profile_ax.set_xlim(-WINDOW / 2, WINDOW / 2)
    profile_ax.set_ylim(-0.05, 1.05)
    profile_ax.set_xlabel('distance from the interface (cell size = 1)')
    profile_ax.set_ylabel('phase field')
    profile_ax.set_title(f'profile across the interface: {label.split(" (")[0]}', loc='left', fontsize=12)
    profile_ax.grid(axis='y', color=GRID, linewidth=0.8)
    profile_ax.legend(frameon=False, fontsize=11, loc='upper left')
fig.suptitle(f'Interface details: window of width {WINDOW:g} of the cell around the first interface on the middle '
             'line (solid = dark, exact pixels)', x=0.02, ha='left', fontsize=15, y=0.97)
fig.savefig(out / 'mesh_independence_2D_interfaces.png', dpi=140, bbox_inches='tight')
print(out / 'mesh_independence_2D_interfaces.png')

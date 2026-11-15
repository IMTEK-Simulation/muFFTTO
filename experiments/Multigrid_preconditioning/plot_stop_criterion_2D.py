"""
Figure of the 2D stopping-test prototype (stop_criterion_2D.py): per target one row with
  left    the four homogenised energies relative to their targets, W_k / W*_k  (1 = on target)
  middle  Poisson ratio nu12 = C1122 / C2222
  right   the two parts of the objective (stiffness misfit, interface penalty)
and vertical lines where the candidate stopping tests would have ended the run.
    python plot_stop_criterion_2D.py     -> figures/stop_criterion_2D.png
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']            # fixed categorical order of the reference palette
STYLES = ['-', '--', '-.', ':']
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
NAMES = ['C1111', 'C1122', 'C2222', 'C1212']
TESTS = [('energies settled (1e-3)', 'settled', 1e-3, '-'), ('energies settled (1e-2)', 'settled', 1e-2, '--'),
         ('objective settled (1e-3)', 'f_settled', 1e-3, ':')]
plt.rcParams.update({'font.size': 13, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False})

files = sorted((HERE / 'logs' / 'stop_criterion').glob('N*_nu*_s*.npz'))
fig, axes = plt.subplots(len(files), 3, figsize=(17, 4.8 * len(files)), squeeze=False)
for row, path in zip(axes, files):
    data = np.load(path)
    iterations = np.arange(1, len(data['f']) + 1)
    scale = path.stem.split('_s')[-1]
    stops = []
    for label, key, tol, style in TESTS:
        hits = np.nonzero(data[key] < tol)[0]
        if len(hits):
            stops.append((label, int(hits[0]) + 1, style))

    ax = row[0]
    ratio = data['energies'] / data['targets']
    for k, name in enumerate(NAMES):
        ax.plot(iterations, ratio[:, k], color=SERIES[k], linestyle=STYLES[k], linewidth=2, label=name)
        ax.annotate(name, (iterations[-1], ratio[-1, k]), xytext=(6, 0), textcoords='offset points', color=INK,
                    va='center', fontsize=12)
    ax.axhline(1.0, color=MUTED, linewidth=1)
    ax.set_ylabel('energy / target energy')
    ax.set_title(f'target scale {scale}: energies relative to the target', loc='left', fontsize=13)
    ax.set_ylim(min(-0.5, float(ratio[5:].min()) - 0.1), max(2.0, float(ratio[5:].max()) + 0.1))

    ax = row[1]
    ax.plot(iterations, data['nu'], color=SERIES[0], linewidth=2)
    ax.axhline(-0.5 / (1 + 0.5), color=MUTED, linewidth=1)            # plane-strain target nu* / (1 - nu*)
    ax.annotate('target', (iterations[-1], -1 / 3), xytext=(6, 0), textcoords='offset points', color=MUTED, va='center')
    ax.set_ylabel('Poisson ratio  C1122 / C2222')
    ax.set_title('Poisson ratio', loc='left', fontsize=13)
    ax.set_ylim(-0.6, 0.6)

    ax = row[2]
    ax.plot(iterations, data['misfit'], color=SERIES[0], linewidth=2, label='stiffness misfit')
    ax.plot(iterations, data['interface'], color=SERIES[1], linestyle='--', linewidth=2, label='interface penalty')
    ax.set_ylabel('part of the objective')
    ax.set_title('the two parts of the objective', loc='left', fontsize=13)
    ax.set_ylim(0, 1.2 * max(float(data['misfit'][10:].max()), float(data['interface'][10:].max())))
    ax.legend(frameon=False, loc='upper right')

    for ax in row:
        for label, iteration, style in stops:
            ax.axvline(iteration, color=INK, linewidth=1.2, linestyle=style)
        ax.set_xlim(0, iterations[-1] * 1.12)
        ax.set_xlabel('iteration')
        ax.grid(axis='y', color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    top = row[0].get_ylim()[1]
    for label, iteration, style in stops:
        row[0].annotate(f'{label}\niteration {iteration}', (iteration, top), xytext=(4, -4), textcoords='offset points',
                        va='top', fontsize=11, color=INK, rotation=90)
    row[0].legend(frameon=False, loc='lower right', ncol=2)
fig.suptitle('2D prototype (128², energy objective): where would a stopping test on the energies end the run?',
             x=0.01, ha='left', fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = HERE / 'figures'
out.mkdir(exist_ok=True)
fig.savefig(out / 'stop_criterion_2D.png', dpi=130)
print(out / 'stop_criterion_2D.png')

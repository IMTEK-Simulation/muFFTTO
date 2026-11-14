"""
Stage-1 overview (64^2): one cell per (method, case).
    number  = CG iterations at the stop (energy estimate, adaptive delay >= 10)
    colour  = solve work / work of the best one-level FFT preconditioner (Green, GJ, dual) of that case
              (teal: cheaper, grey: equal, brown: more expensive; log scale)
Work = K + w_G G + w_T T in fine-level operator applications (serial CPU weights from timing.py).

    python plot_stage1.py [--grid 64] [--w-G 0.2] [--w-T 0.05]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--grid', type=int, default=64)
parser.add_argument('--w-G', type=float, default=0.2)
parser.add_argument('--w-T', type=float, default=0.05)
args = parser.parse_args()

METHODS = [('green', 'Green'), ('green_jacobi', 'Green-Jacobi'), ('dual', 'dual'),
           ('mg', 'MG (Jacobi smoother)'),
           ('mgS-green', 'MG, Green smoother'), ('mgS-green_jacobi', 'MG, GJ smoother'),
           ('mgS-dual', 'MG, dual smoother'),
           ('bnn3-green', 'Green + coarse corr.'), ('bnn3-green_jacobi', 'GJ + coarse corr.'),
           ('bnn3-dual', 'dual + coarse corr.'), ('bnn-green_jacobi', 'GJ + coarse corr. (1 V)'),
           ('add-green_jacobi', 'GJ + coarse (additive)')]
CASES = [('P1', k) for k in (-3, -2, -1, 1, 2, 3)] + [('P2', k) for k in (-3, -2, -1, 1, 2, 3)] + \
        [('P3', k) for k in (-1, -2, -3, -6)]
SUP = {'-': '⁻', '1': '¹', '2': '²', '3': '³', '6': '⁶'}


def load(problem, k, method):
    return np.load(HERE / 'exp_data' / f'{problem}_N{args.grid}_kappa1e{k:+d}_{method}.npz', allow_pickle=True)


it = np.zeros((len(METHODS), len(CASES)), dtype=int)
work = np.zeros_like(it, dtype=float)
for i, (m, _) in enumerate(METHODS):
    for j, (p, k) in enumerate(CASES):
        d = load(p, k, m)
        it[i, j] = int(d['iterations'])
        work[i, j] = float(d['cost_K'] + args.w_G * d['cost_G'] + args.w_T * d['cost_T'])
ratio = work / work[:3].min(axis=0, keepdims=True)

cmap = LinearSegmentedColormap.from_list('teal_grey_brown', ['#0f6e6e', '#5fb3ad', '#e6e6e6', '#d9a066', '#8a4a12'])
norm = LogNorm(vmin=1 / 16, vmax=16)

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(figsize=(15, 7.6))
im = ax.imshow(np.clip(ratio, 1 / 16, 16), cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
for i in range(len(METHODS)):
    for j in range(len(CASES)):
        ax.text(j, i, str(it[i, j]), ha='center', va='center', fontsize=11, color='#1a1a1a')
ax.set_xticks(range(len(CASES)))
ax.set_xticklabels([f'10{"".join(SUP[c] for c in str(k))}' for _, k in CASES])
ax.set_yticks(range(len(METHODS)))
ax.set_yticklabels([lab for _, lab in METHODS])
ax.set_xlabel('phase contrast κ')
ax.tick_params(length=0)
for x in (5.5, 11.5):
    ax.axvline(x, color='white', lw=4)
for y in (2.5, 3.5, 6.5):
    ax.axhline(y, color='white', lw=4)
for x, title in [(2.5, 'P1  square inclusion'), (8.5, 'P2  58 circles (orthotropic)'), (13.5, 'P3  TO design')]:
    ax.text(x, -0.75, title, ha='center', va='bottom', fontsize=13)
for y, title in [(1, 'one-level\nFFT'), (3, 'multigrid'), (5, 'MG with FFT\nsmoother'), (9, 'FFT + MG\ncoarse corr.')]:
    ax.text(len(CASES) - 0.35, y, title, ha='left', va='center', fontsize=11, color='#4d4d4d')
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.13, ticks=[1 / 16, 1 / 4, 1, 4, 16])
cb.ax.set_yticklabels(['1/16', '1/4', '1', '4', '16'])
cb.set_label('work / best one-level FFT preconditioner')
ax.set_title(f'{args.grid}² grid: CG iterations (numbers) and solve work (colour)', pad=34, fontsize=14)
fig.tight_layout()
out = HERE / 'figures'
out.mkdir(exist_ok=True)
fig.savefig(out / f'stage1_overview_N{args.grid}.png', dpi=150)
print(out / f'stage1_overview_N{args.grid}.png')

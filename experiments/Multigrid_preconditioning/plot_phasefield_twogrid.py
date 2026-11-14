"""
Figure of phasefield_twogrid.py:
  left   true relative energy error vs solve work (fine-level K applications + 0.2 Green + 0.05 transfers)
  right  error remaining per wavenumber shell after a stage, relative to the initial error (strain-energy weighted)

    python plot_phasefield_twogrid.py [--eta 2]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullLocator, MaxNLocator

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--eta', type=float, default=2.0)
args = parser.parse_args()
d = np.load(HERE / 'exp_data' / f'phasefield_twogrid_N256_eta{args.eta:g}.npz', allow_pickle=True)
results = d['results'].item()
spectra = np.load(HERE / 'exp_data' / f'phasefield_spectra_N256_eta{args.eta:g}.npz', allow_pickle=True)['spectra'].item()

G_DARK, G_MID, G_LIGHT, G_PALE = '#0b5d1e', '#2ca02c', '#74c476', '#a1d99b'
STYLE = {
    'green': dict(color=G_DARK, ls='-', marker=None, label='Green-PCG'),
    'green_jacobi': dict(color='#1f77b4', ls='-', marker=None, label='Green-Jacobi-PCG'),
    'dual': dict(color='#d62728', ls='-', marker=None, label='dual-PCG'),
    'mg': dict(color='#222222', ls='-', marker=None, label='MG-PCG (Jacobi smoother)'),
    'mgS-green': dict(color=G_MID, ls='--', marker='s', label='MG-PCG, Chebyshev-Green smoother'),
    'green2grid 2,2': dict(color=G_LIGHT, ls=':', marker='o', label='your 2-grid: 2 fine + 2 coarse Green-CG'),
    'green2grid 5,5': dict(color=G_LIGHT, ls='-.', marker='^', label='your 2-grid: 5 fine + 5 coarse Green-CG'),
    'green2grid 5,20': dict(color=G_PALE, ls='--', marker='D', label='your 2-grid: 5 fine + 20 coarse Green-CG'),
}

plt.rcParams.update({'font.size': 13})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.4))

for key, st in STYLE.items():
    if key not in results:
        continue
    w, e = results[key]
    n = len(w)
    ax1.semilogy(w, np.maximum(e, 1e-16), color=st['color'], ls=st['ls'], lw=2, marker=st['marker'],
                 markevery=max(1, n // 8), ms=7, label=f'{st["label"]}  ({w[-1]:.0f})')
ax1.axhline(1e-12, color='#888888', lw=1)
ax1.text(5, 2e-12, 'tolerance', color='#666666', fontsize=11, va='bottom')
ax1.set_xlim(left=0)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.set_ylim(1e-13, 2)
ax1.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
ax1.yaxis.set_major_formatter(LogFormatterMathtext())
ax1.yaxis.set_minor_locator(NullLocator())
ax1.set_xlabel('solve work [fine-grid operator applications]')
ax1.set_ylabel('relative energy error  ||e||²_K / ||e₀||²_K')
ax1.set_title(f'convergence, 256² phase-field TO design, η = {args.eta:g}h')
ax1.legend(fontsize=10.5, loc='upper right', title='(work to tolerance)', title_fontsize=10.5)

k = np.arange(len(spectra['initial']))
init = spectra['initial']
SPEC = [('Green-CG 2', G_PALE, '-', None), ('Green-CG 5', G_LIGHT, '-', None), ('Green-CG 20', G_DARK, '-', None),
        ('2-grid 5,20: after coarse', '#e08214', '--', 's'), ('MG V-cycle 1', '#222222', '-.', '^')]
for key, color, ls, marker in SPEC:
    if key not in spectra:
        continue
    lab = (key.replace('Green-CG', 'after Green-CG').replace('MG V-cycle 1', 'after 1 MG V-cycle (Jacobi smoother)')
           .replace(': after coarse', ': 5 fine Green-CG + coarse correction'))
    ax2.semilogy(k[1:], spectra[key][1:] / init[1:], color=color, ls=ls, lw=2, marker=marker, markevery=12, ms=7,
                 label=lab.replace('2-grid 5,20', 'your 2-grid (5, 20)'))
ax2.axvline(64, color='#888888', lw=1)
ax2.text(65, 3, 'coarse-grid\nNyquist (128²)', color='#666666', fontsize=11, va='top')
ax2.set_xlim(0, 182)
ax2.axhline(1, color='#888888', lw=1)
ax2.set_ylim(1e-5, 1e3)
ax2.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
ax2.yaxis.set_major_formatter(LogFormatterMathtext())
ax2.yaxis.set_minor_locator(NullLocator())
ax2.set_xlabel('wavenumber |k| (shell)')
ax2.set_ylabel('energy error left / initial (per |k| shell)')
ax2.set_title('which frequencies each stage removes (energy norm)')
ax2.legend(fontsize=10.5, loc='lower right')
fig.tight_layout()
out = HERE / 'figures' / f'phasefield_twogrid_N256_eta{args.eta:g}.png'
fig.savefig(out, dpi=150)
print(out)

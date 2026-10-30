"""Summary of run_eyre_milton.py: cost (FFT pairs = iterations x FFT pairs per iteration) to the true relative energy
error 1e-6 over the contrast, one panel per problem, Green-PCG, dual-PCG and Eyre-Milton (two references)
-> figures/eyre_milton_vs_contrast.png and a markdown table on stdout.

    python plot_eyre_milton.py
"""
import glob
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
METHODS = [  # key, label, colour, line style, marker, filled
    ('green', 'Green-PCG (1 FFT/it)', '#2a78d6', '-', 'o', True),
    ('dual', 'dual-PCG (2 FFT/it)', '#eb6834', '-', 's', True),
    ('em_c0_matrix', r'Eyre-Milton, $c_0$ = matrix (1 FFT/it)', '#0f7a54', '-', '^', True),
    ('em_c0_opt', r'Eyre-Milton, $c_0=\sqrt{g_{min} g_{max}}$ (1 FFT/it)', '#5fd3a6', '--', 'v', False),
]

runs = {}
for f in sorted(glob.glob(str(HERE / 'exp_data_eyre_milton' / '*_N256_*.npz'))):
    d = np.load(f, allow_pickle=True)
    runs[(str(d['problem']), float(d['kappa']))] = d['results'].item()

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, problem in zip(axes, ['P1', 'P2', 'P3']):
    kappas = sorted(k for p, k in runs if p == problem)
    for key, label, colour, ls, marker, filled in METHODS:
        pts = [(k, runs[(problem, k)][key]['k_true'] * runs[(problem, k)][key]['nb_fft']) for k in kappas
               if runs[(problem, k)][key]['k_true'] >= 0]
        if pts:
            ax.plot(*zip(*pts), ls=ls, marker=marker, color=colour, lw=2, ms=7,
                    mfc=colour if filled else 'white', label=label)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(problem)
    ax.set_xlabel(r'contrast $\kappa$')
    ax.grid(True, color='0.92', which='both')
axes[0].set_ylabel('FFT pairs to relative energy error $10^{-6}$')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1))
(HERE / 'figures').mkdir(exist_ok=True)
fig.savefig(HERE / 'figures' / 'eyre_milton_vs_contrast.png', dpi=150)

print('| problem | kappa | Green k (prod) | dual k (prod) | EM c0=1 k | EM c0=opt k | cost Green / dual / EM1 / EMopt |'
      ' EM rate obs/bound c0=1 | c0=opt |')
print('|---|---|---|---|---|---|---|---|---|')
for (problem, kappa), r in sorted(runs.items()):
    cost = ' / '.join(str(r[m]['k_true'] * r[m]['nb_fft']) if r[m]['k_true'] >= 0 else '-' for m, *_ in METHODS)
    print(f'| {problem} | {kappa:g} | {r["green"]["k_true"]} ({r["green"]["k_prod"]}) | '
          f'{r["dual"]["k_true"]} ({r["dual"]["k_prod"]}) | {r["em_c0_matrix"]["k_true"]} | '
          f'{r["em_c0_opt"]["k_true"]} | {cost} | '
          f'{r["em_c0_matrix"]["rate_observed"]:.3f}/{r["em_c0_matrix"]["rate_bound"]:.3f} | '
          f'{r["em_c0_opt"]["rate_observed"]:.3f}/{r["em_c0_opt"]["rate_bound"]:.3f} |')

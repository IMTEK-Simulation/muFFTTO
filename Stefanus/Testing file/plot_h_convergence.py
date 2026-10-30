"""
h-convergence of the Composite A stress-strain curves (1 x 2 cell, finite strain, trust-region Newton-CG):
Q1-SRI at NX = 64 / 128 / 256 pixels per primitive cell, Q1 F-bar at 64 / 128 for comparison, paper Fig. 4.
Reads the per-increment frames (exp_data/frames_TR_*); writes figures/h_convergence_TR.png and prints the
buckling point (first increment whose secant stiffness drops below half of the initial one) of every run.
"""
import glob
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = [  # (label, frame-folder tag, colour, line style, marker)
    ('SRI, NX = 64', '64x110_cells1x2_sri', '#9fd49b', '-', 'o'),
    ('SRI, NX = 128', '128x222_cells1x2_sri_gpu', '#3a9a3a', '-', 's'),
    ('SRI, NX = 256', '256x444_cells1x2_sri_gpu', '#0b4d0b', '-', 'D'),
    ('F-bar, NX = 64', '64x110_cells1x2_fbar', '#c3a6e0', '--', 'o'),
    ('F-bar, NX = 128', '128x222_cells1x2_fbar_gpu', '#6a3d9a', '--', 's'),
]
fig4 = np.genfromtxt(os.path.join(HERE, 'exp_data', 'paper_fig4_compositeA.csv'), delimiter=',', skip_header=2,
                     names=True, dtype=None, encoding=None)
EPS_CR = {'X': 0.016, 'Y': 0.026}


def curve(direction, tag):
    fs = sorted(glob.glob(os.path.join(HERE, 'exp_data', f'frames_TR_{direction}_{tag}', 'inc*.npz')))
    return (np.r_[0.0, [float(np.load(f)['eps']) for f in fs]],
            np.r_[0.0, [float(np.load(f)['stress_MPa']) for f in fs]])


plt.rcParams.update({'font.size': 13})
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), layout='constrained')
for ax, d in zip(axes, ('X', 'Y')):
    sel = fig4['direction'] == d
    for c, col, ls, lab in (('periodic', 'tab:red', '-.', 'paper Fig. 4: simulation, periodic'),
                            ('experiment', 'k', '-', 'paper Fig. 4: experiment')):
        m = sel & (fig4['curve'] == c)
        ax.plot(fig4['eps'][m], fig4['stress'][m], color=col, ls=ls, lw=2, label=lab)
    ax.axvline(EPS_CR[d], color='tab:red', ls=':', lw=1.2)
    print(f'--- {d}: paper eps_cr = {EPS_CR[d]}')
    for lab, tag, col, ls, mk in RUNS:
        e, s = curve(d, tag)
        if e.size < 2:
            continue
        ax.plot(e, s, ls=ls, color=col, marker=mk, ms=5, lw=2, label=f'muFFTTO {lab} (to $\\varepsilon$ = {e[-1]:.3f})')
        k0 = s[1] / e[1]
        tang = np.diff(s) / np.diff(e)                     # tang[j]: segment e[j] -> e[j + 1]
        b = np.flatnonzero(tang < 0.5 * k0)
        txt = (f'buckles between eps {e[b[0]]:.4f} and {e[b[0] + 1]:.4f}, stress {s[b[0]]:.5f}-{s[b[0] + 1]:.5f} MPa'
               if b.size else 'not buckled yet')
        print(f'{lab:<16} initial slope {k0:.3f} MPa | {txt} | last eps {e[-1]:.3f}, stress {s[-1]:.5f} MPa')
    ax.set(xlabel=f'nominal strain $\\varepsilon_{d}$', ylabel='nominal stress [MPa]', xlim=(0, 0.205), ylim=(0, None),
           title=f'Composite A, compression in {d} (1 x 2 cell, lateral side free)')
    ax.grid(alpha=0.3)
handles, labels = axes[1].get_legend_handles_labels()
axes[0].legend(frameon=False, fontsize=10.5, loc='upper left')
axes[1].legend(frameon=False, fontsize=10.5, loc='lower right')
os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
out = os.path.join(HERE, 'figures', 'h_convergence_TR.png')
fig.savefig(out, dpi=120)
print(out)

"""
Stress-strain response from homogenized_stress.csv of
stefanus_2D_homogenization_finite_strain_elasticity_NeoHookean_trust_region.py, against Li & Rudykh (2019), Fig. 4.

The loaded direction is the diagonal component of F - I that changes most (x: F_00, y: F_11); the nominal strain is
eps = 1 - F_kk and the nominal stress -<P_kk>. The other diagonal stress <P_ll> is plotted next to it: the paper's test
has a stress-free lateral side (<P_ll> = 0); with F_ll prescribed it is not zero.

usage:  python plot_homogenized_stress.py [run_1/homogenized_stress.csv run_2/homogenized_stress.csv ...]
        (several runs: drawn together, labelled by their folder name; the figure is saved next to the first)
"""
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

MU_M_MPA = 0.6      # matrix shear modulus [MPa]: the CSV holds <P> in units of mu_m

here = os.path.dirname(os.path.abspath(__file__))
csv_paths = sys.argv[1:] or sorted(glob.glob(os.path.join(          # no arguments: all runs found in exp_data
    here, 'exp_data', 'stefanus_2D_homogenization_finite_strain_elasticity_NeoHookean_trust_region',
    '*', 'homogenized_stress.csv')))[:4]
if not csv_paths:
    sys.exit('no homogenized_stress.csv found: run the trust-region script first, or pass the CSV paths')
paper_path = os.path.join(here, '..', '..', 'Testing file', 'exp_data', 'paper_fig4_compositeA.csv')
run_styles = [('tab:green', 'o', '-'), ('tab:blue', 's', '--'), ('tab:purple', '^', ':'), ('tab:orange', 'v', '-.')]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.6), layout='constrained')

for csv_path, (color, marker, style) in zip(csv_paths, run_styles):
    data = np.atleast_1d(np.genfromtxt(csv_path, delimiter=',', names=True))   # 1 increment -> still 1 row
    F_diag = np.stack([data['F_00'], data['F_11']])
    k = int(np.argmax(np.abs(F_diag[:, -1] - 1.0)))      # loaded axis
    l = 1 - k                                            # lateral axis
    axis = 'XY'[k]
    eps = np.r_[0.0, 1.0 - F_diag[k]]
    stress = np.r_[0.0, -data[f'P_{k}{k}'] * MU_M_MPA]
    lateral_stress = np.r_[0.0, data[f'P_{l}{l}'] * MU_M_MPA]
    label = os.path.basename(os.path.dirname(os.path.abspath(csv_path)))   # e.g. Nx=128Ny=111_Green_bilinear_rectangle
    side = 'free sides (void layer)' if '_void' in label else 'lateral F prescribed'
    ax1.plot(eps, stress, color=color, marker=marker, ls=style, ms=3, lw=1.8, label=f'{label} ({side})')
    ax2.plot(eps, lateral_stress, color=color, marker=marker, ls=style, ms=3, lw=1.8, label=label)

if os.path.exists(paper_path):
    paper = np.genfromtxt(paper_path, delimiter=',', skip_header=2, names=True, dtype=None, encoding=None)
    for curve, color, style, label in (('periodic', 'tab:red', '-.', 'Li & Rudykh, simulation'),
                                       ('experiment', 'k', '-', 'Li & Rudykh, experiment')):
        m = (paper['direction'] == axis) & (paper['curve'] == curve)
        ax1.plot(paper['eps'][m], paper['stress'][m], color=color, ls=style, lw=1.6,
                 label=f'{label} (lateral side stress free)')
ax1.set(xlabel=f'nominal strain $\\varepsilon_{axis}$', ylabel='nominal stress $-\\langle P_{%s%s} \\rangle$ [MPa]'
        % (axis.lower(), axis.lower()), title=f'(a) compression in {axis}')
ax1.legend(frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(0.0, -0.18))

ax2.legend(frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(0.0, -0.18))
ax2.axhline(0.0, color='0.5', lw=0.8)
other = 'XY'[l].lower()
ax2.set(xlabel=f'nominal strain $\\varepsilon_{axis}$',
        ylabel=f'lateral stress $\\langle P_{{{other}{other}}} \\rangle$ [MPa]',
        title='(b) lateral stress (0 in the paper\'s test)')

for ax in (ax1, ax2):
    ax.grid(alpha=0.3)
out = os.path.join(os.path.dirname(os.path.abspath(csv_paths[0])), 'stress_strain.png')
fig.savefig(out, dpi=150)
print(out)

"""
Slide 'Phase-field regularization': the double-well potential W(rho) = rho^2 (1 - rho)^2 of F_pf, minima at rho = 0
(void) and rho = 1 (solid). Same style and width as pf_profile (plot_phase_field_panels.py).
    python plot_double_well.py  ->  figures/pf_double_well.{pdf,png} and the talk figures/
"""
import os
import shutil

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})

fig, b = plt.subplots(figsize=(4.8, 2.4))
r = np.linspace(-0.12, 1.12, 1001)
W = r ** 2 * (1 - r) ** 2
b.plot(r, W, color='k', lw=3)
b.plot([0, 1], [0, 0], 'o', color='tab:red', ms=9, zorder=3)
b.text(0, 0.026, 'void', color='tab:red', ha='center', va='bottom', fontsize=13)
b.text(1, 0.026, 'solid', color='tab:red', ha='center', va='bottom', fontsize=13)
b.set_title('double well potential', fontsize=14)
b.set_xlim(-0.12, 1.12), b.set_ylim(-0.006, 0.085)
b.set_xticks([0, 0.5, 1], ['$0$', '$0.5$', '$1$']), b.set_yticks([0, 0.05])
b.set(xlabel=r'$\rho$', ylabel=r'$\rho^2(1-\rho)^2$')
b.grid(alpha=0.3)
for ext in ('pdf', 'png'):
    p = os.path.join(HERE, 'figures', f'pf_double_well.{ext}')
    fig.savefig(p, dpi=200, bbox_inches='tight', pad_inches=0.02)
    if os.path.isdir(TALK):
        shutil.copy(p, TALK)
print('wrote pf_double_well (.pdf/.png) to figures/ and', TALK)

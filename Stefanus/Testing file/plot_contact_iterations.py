"""
Solver cost over the load path of the contact test (run_tm_contact_64.sh) and the 256 px true-hole run: per load
increment the trust-region Newton steps, the CG iterations and the CG iterations per Newton step, all with Green
(simplified material: inclusion contrast 10, Lambda/mu 100). Reads the run logs (runs may still be going).

    python plot_contact_iterations.py  -> figures/contact_iterations.png
"""
import os
import re

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = [  # (label, log, colour, marker)
    ('64 px, true holes', 'log_contact64_holes', '#7a7a7a', 'o'),
    ('256 px, true holes', 'log_simplify256_c10_L100', '#303030', 'D'),
    ('64 px, third medium k = 1e-3', 'log_contact64_TM1e-3', '#e08a2c', 's'),
    ('64 px, third medium k = 1e-4', 'log_contact64_TM1e-4', '#c0392b', '^'),
]


def increments(log):
    """eps, TR Newton steps, CG iterations per load increment"""
    rows = re.findall(r'^inc +\d+ +eps_X = ([\d.]+).*\(TR its (\d+), CG its (\d+)\)',
                      open(os.path.join(HERE, 'exp_data', log + '.txt')).read(), flags=re.M)
    return (np.array([float(r[i]) for r in rows]) for i in range(3))


plt.rcParams.update({'font.size': 11})
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout='constrained')
for label, log, colour, marker in RUNS:
    eps, tr, cg = increments(log)
    for ax, y in zip(axes, (tr, cg, cg / tr)):
        ax.plot(eps, y, color=colour, marker=marker, ms=4, lw=1.6, label=label)
    print(f'{label:30s} to eps {eps[-1]:.3f}: {len(eps)} increments, TR {int(tr.sum())}, CG {int(cg.sum())}, '
          f'CG per Newton step {cg.sum() / tr.sum():.0f}')
for ax, title in zip(axes, ('trust-region Newton steps per load increment', 'CG iterations per load increment',
                            'CG iterations per Newton step')):
    ax.axvline(0.016, color='tab:red', ls=':', lw=1)
    ax.set(xlabel='nominal strain $\\varepsilon_X$ (load increment 0.005)', title=title, xlim=(0, 0.205))
    ax.grid(alpha=0.3)
axes[1].set_yscale('log')
axes[2].set_yscale('log')
axes[0].text(0.017, axes[0].get_ylim()[1] * 0.95, 'buckling', color='tab:red', fontsize=9, va='top')
axes[0].legend(frameon=False, fontsize=9.5)
fig.suptitle('Composite A with Green (CG to energy error 1e-6, adaptive delay >= 10): cost over the load path',
             fontsize=12)
out = os.path.join(HERE, 'figures', 'contact_iterations.png')
fig.savefig(out, dpi=130)
print(f'saved {out}')

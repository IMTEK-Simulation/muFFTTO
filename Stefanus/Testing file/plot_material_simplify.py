"""
Can the material of Composite A be simplified? X compression of the 1 x 2 cell with true holes (run_material_simplify.sh,
SRI Q1): 32 px: inclusion contrast {10, 100} x Lambda/mu {10, 100} and the paper's (1000, 1000); 128 px: contrast 10
with Lambda/mu 100 and 10 (run_material_simplify_128.sh), against paper Fig. 4
and the earlier SRI runs with the paper's material at NX = 64 / 128 / 256. Reads the run logs (the baseline may still be
running) and the frames of the earlier runs; writes figures/material_simplify_X.png.
"""
import glob
import os
import re

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
fig4 = np.genfromtxt(os.path.join(HERE, 'exp_data', 'paper_fig4_compositeA.csv'), delimiter=',', skip_header=2,
                     names=True, dtype=None, encoding=None)
EARLIER = [  # (label, frame-folder tag, colour, marker): paper material, NX pixels per primitive cell
    ('SRI NX = 64', '64x110_cells1x2_sri', '#b8b8b8', 'o'),
    ('SRI NX = 128', '128x222_cells1x2_sri_gpu', '#606060', 's'),
    ('SRI NX = 256', '256x444_cells1x2_sri_gpu', '#303030', 'D'),
]
NEW = [  # (label, log name, colour (hue: inclusion contrast; grid: shade), line style (Lambda/mu), marker)
    ('32 px: contrast 1000, $\\Lambda/\\mu$ 1000 (paper)', 'simplify_c1000_L1000', 'tab:red', '-', 'o'),
    ('32 px: contrast 100, $\\Lambda/\\mu$ 100', 'simplify_c100_L100', 'tab:blue', '-', 's'),
    ('32 px: contrast 100, $\\Lambda/\\mu$ 10', 'simplify_c100_L10', 'tab:blue', '--', 's'),
    ('32 px: contrast 10, $\\Lambda/\\mu$ 100', 'simplify_c10_L100', '#f5b46a', '-', '^'),
    ('32 px: contrast 10, $\\Lambda/\\mu$ 10', 'simplify_c10_L10', '#f5b46a', '--', '^'),
    ('128 px: contrast 10, $\\Lambda/\\mu$ 100', 'simplify128_c10_L100', '#c25e00', '-', 'v'),
    ('128 px: contrast 10, $\\Lambda/\\mu$ 10', 'simplify128_c10_L10', '#c25e00', '--', 'v'),
]


def from_frames(tag):
    fs = sorted(glob.glob(os.path.join(HERE, 'exp_data', f'frames_TR_X_{tag}', 'inc*.npz')))
    return (np.r_[0.0, [float(np.load(f)['eps']) for f in fs]],
            np.r_[0.0, [float(np.load(f)['stress_MPa']) for f in fs]])


def from_log(name):
    """(eps, nominal stress [MPa], CG its) per increment from exp_data/log_<name>.txt"""
    rows = re.findall(r'^inc +\d+ +eps_X = ([\d.]+).*?nominal stress ([\d.e+-]+) MPa.*CG its (\d+)\)',
                      open(os.path.join(HERE, 'exp_data', f'log_{name}.txt')).read(), flags=re.M)
    e, s, cg = (np.array([float(r[i]) for r in rows]) for i in range(3))
    return np.r_[0.0, e], np.r_[0.0, s], cg


plt.rcParams.update({'font.size': 13})
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), layout='constrained')
for ax, xmax in zip(axes, (0.205, 0.052)):
    for c, col, ls, lab in (('periodic', 'tab:red', '-.', 'paper Fig. 4: simulation, periodic'),
                            ('experiment', 'k', ':', 'paper Fig. 4: experiment')):
        m = (fig4['direction'] == 'X') & (fig4['curve'] == c)
        ax.plot(fig4['eps'][m], fig4['stress'][m], color=col, ls=ls, lw=2.5, label=lab)
    for lab, tag, col, mk in EARLIER:
        e, s = from_frames(tag)
        ax.plot(e, s, color=col, marker=mk, ms=4, lw=1.5, label=f'earlier: paper material, {lab}')
    for lab, tag, col, ls, mk in NEW:
        e, s, cg = from_log(tag)
        ax.plot(e, s, color=col, ls=ls, marker=mk, ms=6, lw=2, mfc='white' if ls == '--' else col, label=lab)
        if xmax > 0.1:
            print(f'{tag:21s}: ' + ', '.join(f'{ei:.3f}: {si:.5f}' for ei, si in zip(e[1:], s[1:]))
                  + f' MPa | CG per increment {int(cg.mean())}')
    ax.set(xlabel='nominal strain $\\varepsilon_X$', ylabel='nominal stress [MPa]', xlim=(0, xmax), ylim=(0, None))
    ax.grid(alpha=0.3)
axes[0].set_title('Composite A, compression in X (1 x 2 cell, true holes)')
axes[1].set_title('zoom: up to and through buckling')
axes[0].legend(frameon=False, fontsize=10, loc='upper left')
os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
out = os.path.join(HERE, 'figures', 'material_simplify_X.png')
fig.savefig(out, dpi=130)
print(f'saved {out}')

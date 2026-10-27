"""
CG cost of the Hertz runs (regular grid, Green-Jacobi, energy stop 1e-6 / delay 10) against the third-medium stiffness
K_V and the mesh: CG iterations per linear solve (per trust-region Newton iteration), before contact (P < 0.01) and in
contact, and the total CG iterations of the run.

usage:  python plot_cg_study.py      -> cg_study.png
"""
import glob
import os
import re
import shutil
import tempfile

import numpy as np
import netCDF4
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

HERE = os.path.dirname(os.path.realpath(__file__))
MESHES = [32, 64, 128]
KVS = ['1e-4', '1e-5', '1e-6']
# one hue (blue) for the method, stiffness = shade + line style + marker (light = stiff medium)
STYLE = {'1e-4': dict(color='#86b6ef', ls=(0, (1, 1.5)), marker='s', mfc='white'),
         '1e-5': dict(color='#3d7fd0', ls=(0, (4, 2)), marker='D', mfc='white'),
         '1e-6': dict(color='#184f95', ls='-', marker='o', mfc='#184f95')}
PLAIN = FuncFormatter(lambda v, _: f'{v:.0f}')


def load(n, kv):
    runs = [r for r in glob.glob(os.path.join(HERE, 'runs', f'hertz_R1_gap0.02_kv{kv}_a1e-7_push0.035_Q1_regular_N{n}_*'))
            if '_restart' not in r]
    if not runs:
        return None
    tmp = os.path.join(tempfile.mkdtemp(), 'c.nc')
    shutil.copy(os.path.join(max(runs, key=os.path.getmtime), 'contact_fracture.nc'), tmp)
    try:
        with netCDF4.Dataset(tmp) as ds:
            r = {k: np.asarray(ds[k][:]) for k in ('nb_hessp', 'nb_outer', 'crown_force')}
    except (OSError, KeyError):
        return None
    return r if len(r['crown_force']) else None


fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), layout='constrained')
for kv in KVS:
    rows = []
    for n in MESHES:
        r = load(n, kv)
        if r is None:
            continue
        h, o, P = r['nb_hessp'], r['nb_outer'], r['crown_force']
        per = lambda m: h[m].sum() / o[m].sum() if o[m].sum() else np.nan
        rows.append((n, per((P < 0.01) & (o > 0)), per((P >= 0.01) & (o > 0)), h.sum(), len(P) == 35))
    if not rows:
        continue
    a = np.array(rows, dtype=float)
    for ax, col in zip(axes, (1, 2, 3)):
        ax.plot(a[:, 0], a[:, col], lw=2, ms=8, label=f'$K_V$ = {kv}', **STYLE[kv])
        for n, v, done in zip(a[:, 0], a[:, col], a[:, 4]):
            if not done:  # unfinished run: marked with a ring
                ax.plot(n, v, 'o', ms=15, mfc='none', mec=STYLE[kv]['color'], mew=1.2)
for ax, title in zip(axes, ('CG per solve before contact (P < 0.01)', 'CG per solve in contact',
                            'CG iterations, whole run (35 load steps)')):
    ax.set_xscale('log', base=2)
    ax.set_xticks(MESHES, [str(m) for m in MESHES])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('pixels per side (regular grid)')
    ax.set_ylabel('CG iterations')
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(PLAIN)
    ax.grid(True, color='0.9')
    ax.set_title(title, fontsize=11)
axes[0].legend(frameon=False, fontsize=10, title='third-medium stiffness / E\n(ring: run not finished)', title_fontsize=9)
fig.suptitle('Hertz test, regular grid, Green-Jacobi: CG cost against the third-medium contrast and the mesh '
             '(energy stop 1e-6, delay 10; bound ~ $\\frac{1}{2}\\sqrt{1/K_V}\\,\\ln(2/10^{-6})$ = 730 / 2300 / 7300)',
             fontsize=12)
out = os.path.join(HERE, 'cg_study.png')
fig.savefig(out, dpi=120)
print(f'wrote {out}')

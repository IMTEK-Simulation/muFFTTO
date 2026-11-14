"""
Slide figure for 'Phase-field fracture (AT2), 1024^2': polydisperse pores, l = 1/32 (l/h = 32), Green (NEMO, 2026-11-07)
vs dual (laptop GPU, 2026-10-19/20, with a restart segment). Against the load step:
  top     macroscopic stress (from the logs); shaded: cracked (max d > 0.95)
  bottom  mean CG iterations per elasticity solve in the load step
    python plot_slide_AT2_1024.py  ->  figures/slide_AT2_1024.{png,pdf} (+ talk figures/)
"""
import csv
import os
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
STEM = 'AT2_N1024_{}_polydisperse_pores_l0.03125'


def per_step(P):
    files = [os.path.join(HERE, 'exp_data', STEM.format(P) + '_iterations.csv')]
    r2 = os.path.join(HERE, 'exp_data', STEM.format(P) + '_restart_iterations.csv')
    if os.path.exists(r2):
        files.append(r2)
    d = {}
    for f in files:                                     # later file (restart) overrides repeated steps
        rows = list(csv.DictReader(open(f)))
        steps = {int(r['step']) for r in rows}
        for s in steps:
            d[s] = [int(r['cg_u']) for r in rows if int(r['step']) == s]
    s = np.array(sorted(d))
    return s, np.array([np.mean(d[k]) for k in s])


def log_curve(P):
    stress, maxd = {}, {}
    for suf in ('', '_restart'):
        f = os.path.join(HERE, 'logs', STEM.format(P) + suf + '.log')
        if not os.path.exists(f):
            continue
        for m in re.finditer(r'^step\s+(\d+)\s+E_xx = \S+\s+sigma_xx = (\S+)\s+max d = (\S+)', open(f).read(), re.M):
            stress[int(m.group(1))], maxd[int(m.group(1))] = float(m.group(2)), float(m.group(3))
    s = np.array(sorted(stress))
    return s, np.array([stress[k] for k in s]), np.array([maxd[k] for k in s])


plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.6, 4.6), sharex=True, layout='constrained',
                             gridspec_kw=dict(height_ratios=[0.7, 1]))
s, sig, md = log_curve('dual')
crack = s[np.argmax(md > 0.95)]
for ax in (a1, a2):
    ax.axvspan(crack, s.max(), color='0.9', zorder=0)
    ax.grid(alpha=0.3)
a1.plot(s, sig * 1e3, 'k-', lw=2)                       # numbers as the run's stress x 1e3, axis label requested as sigma / mu
a1.set(ylabel=r'$\bar\sigma / \mu$'), a1.set_ylim(0, None)
a1.text(crack + 1, 0.12 * a1.get_ylim()[1], 'cracked', fontsize=11)
for P, lab, col in (('Green', 'Green', 'tab:green'), ('dual', r'dual ($\times 2$)', 'tab:red')):
    st, cg = per_step(P)
    cg = cg * (2 if P == 'dual' else 1)
    a2.plot(st, cg, '-', color=col, lw=2, label=lab)
    print(P, 'before', round(cg[st < crack].mean(), 1), 'cracked', round(cg[st >= crack].mean(), 1))
a2.set(xlabel='load step', ylabel='CG iterations per solve'), a2.set_ylim(0, None), a2.set_xlim(0, s.max())
a2.legend(loc='upper left', fontsize=11, framealpha=0.95)
out = os.path.join(HERE, 'figures', 'slide_AT2_1024')
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=150)
    shutil.copy(f'{out}.{ext}', TALK)
print('wrote', out, 'crack at step', crack)

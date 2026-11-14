"""
Slide figure next to the cracked 1024^2 cell (polydisperse pores, l = 1/32), same style and size as plot_slide_AT2_1024.py:
  top     macroscopic stress over macroscopic strain, dual runs 32^2 ... 1024^2 (logs; stress x 1e3 labelled sigma / mu as
          on the slide_AT2_1024 figure), 1024^2 black, coarser grids grey
  bottom  mean elastic CG iterations per solve over the whole run vs grid size: Green, Green-Jacobi, dual (x 2)
          (the same numbers as the _study video panel: finished runs only, restart segments joined)
    python plot_slide_AT2_1024_study.py  ->  figures/slide_AT2_1024_study.{png,pdf} (+ talk figures/)
    NO_DUAL=1: Green and Green-Jacobi only (the dual comes later in the talk) -> slide_AT2_1024_study_nodual.*
    CG_ONLY=1: only the bottom panel (CG per solve vs grid size), 4.6 x 2.9 in -> slide_AT2_1024_study_cg.*
"""
import os
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
STEM = 'AT2_N{}_{}_polydisperse_pores_l0.03125'
GRIDS = (32, 64, 128, 256, 512, 1024)
NO_DUAL = os.environ.get('NO_DUAL', '0') == '1'
CG_ONLY = os.environ.get('CG_ONLY', '0') == '1'


def log_curve(n, p):
    """load and sigma_xx per load step from the log(s), the restart segment overriding repeated steps"""
    strain, stress = {}, {}
    for suf in ('', '_restart'):
        f = os.path.join(HERE, 'logs', STEM.format(n, p) + suf + '.log')
        if not os.path.exists(f):
            continue
        for m in re.finditer(r'^step\s+(\d+)\s+E_xx = (\S+)\s+sigma_xx = (\S+)', open(f).read(), re.M):
            strain[int(m.group(1))], stress[int(m.group(1))] = float(m.group(2)), float(m.group(3))
    s = sorted(stress)
    return np.array([strain[k] for k in s]), np.array([stress[k] for k in s])


def cg_per_solve(n, p):
    """mean elastic CG iterations per solve of a finished run (as make_AT2_video.cg_per_solve), None if missing"""
    name, log = (os.path.join(HERE, d, STEM.format(n, p)) for d in ('exp_data', 'logs'))
    if not os.path.exists(name + '_iterations.csv'):
        return None
    restarted = os.path.exists(name + '_restart_iterations.csv')
    if 'Elapsed time' not in open(log + ('_restart' if restarted else '') + '.log').read():
        return None
    it = np.genfromtxt(name + '_iterations.csv', delimiter=',', names=True, ndmin=1)
    if restarted:
        r = np.genfromtxt(name + '_restart_iterations.csv', delimiter=',', names=True, ndmin=1)
        it = np.concatenate([it[it['step'] < r['step'].min()], r])
    return it['cg_u'].mean()


plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
if CG_ONLY:
    fig, a2 = plt.subplots(figsize=(4.6, 2.9), layout='constrained')
    a1 = None
else:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.6, 4.6), layout='constrained', gridspec_kw=dict(height_ratios=[0.8, 1]))
for ax in (a1, a2):
    if ax is not None:
        ax.grid(alpha=0.3)
if not CG_ONLY:
    coarse = [n for n in GRIDS[:-1] if os.path.exists(os.path.join(HERE, 'logs', STEM.format(n, 'dual') + '.log'))]
    for n, grey in zip(coarse, np.linspace(0.8, 0.5, len(coarse))):
        e, sig = log_curve(n, 'dual')
        a1.plot(np.r_[0, e], np.r_[0, sig] * 1e3, color=str(grey), lw=1.3)
    e, sig = log_curve(1024, 'dual')
    a1.plot(np.r_[0, e], np.r_[0, sig] * 1e3, 'k-', lw=2, label='$1024^2$')
    a1.plot([], [], color='0.65', lw=1.3, label=f'${coarse[0]}^2$--${coarse[-1]}^2$')
    a1.set(xlabel=r'macroscopic strain $\bar\varepsilon_{xx}$', ylabel=r'$\bar\sigma / \mu$')
    a1.set_xlim(0, e.max()), a1.set_ylim(0, None)
    a1.legend(loc='lower center', fontsize=11, framealpha=0.95)
for p, lab, col in (('Green', 'Green', 'tab:green'), ('Green_Jacobi', 'Green-Jacobi', 'tab:blue'),
                    ('dual', r'dual ($\times 2$)', 'tab:red'))[:2 if NO_DUAL else 3]:
    pts = [(n, cg_per_solve(n, p)) for n in GRIDS]
    pts = np.array([(n, v * (2 if p == 'dual' else 1)) for n, v in pts if v is not None])
    a2.plot(pts[:, 0], pts[:, 1], '-o', color=col, lw=2, ms=5, label=lab)
    print(p, dict(zip(pts[:, 0].astype(int), np.round(pts[:, 1]).astype(int))))
a2.set_xscale('log', base=2)
a2.set_xticks(GRIDS, [f'${n}^2$' for n in GRIDS]), a2.minorticks_off()
a2.set(xlabel='grid size', ylabel='CG iterations per solve' if NO_DUAL or CG_ONLY else r'CG per solve (dual $\times 2$)'), a2.set_ylim(0, None)   # CG_ONLY: x2 stated in the legend
a2.legend(loc='upper left', fontsize=11, framealpha=0.95)
out = os.path.join(HERE, 'figures', 'slide_AT2_1024_study' + ('_nodual' if NO_DUAL else '') + ('_cg' if CG_ONLY else ''))
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=150)
    shutil.copy(f'{out}.{ext}', TALK)
print('wrote', out)

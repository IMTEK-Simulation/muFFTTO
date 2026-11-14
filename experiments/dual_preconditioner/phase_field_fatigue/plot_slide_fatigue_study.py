"""Slide figure (right half of the fatigue slide), same style and size as ../damage_AT2_small_strain/plot_slide_AT2_1024_study.py:
  top     CG per elastic solve during crack growth vs grid size, 2D 64^2 ... 1024^2 (definition and data as
          plot_slide_fatigue.py -> fatigue_cg_vs_grid_2D), dual x 2, grey: l/h < 2
  bottom  cumulative wall time (elastic + damage solves) of the 1024^2 runs over the load cycles, final value labelled
          (as the wall-time panel of make_FAT_slide_video.py, last frame)
    python plot_slide_fatigue_study.py  ->  figures/slide_fatigue_study.{png,pdf} (+ talk figures/)
"""
import glob
import os
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
TAG = '_incl11.6667_l0.015625_k0.0001_nemo2D_precrack_asymptotic_amp0.002_cyc1500'
GRIDS = [64, 128, 256, 512, 1024]
PRECS = [('Green', 'Green', 'tab:green', 'o'), ('Green_Jacobi', 'Green-Jacobi', 'tab:blue', 's'),
         ('dual', r'dual ($\times 2$)', 'tab:red', '^')]


def load(N, P):
    """cycles csv of the 2D NEMO run, restart segments (_r<k>) joined (as plot_slide_fatigue.load)"""
    files = glob.glob(os.path.join(HERE, 'exp_data', f'FAT_N{N}_{P}_incl*_nemo2D*_precrack_*_cycles.csv'))
    files = [f for f in files if re.search(r'_nemo2D(_r\d+)?_precrack', f)]
    files.sort(key=lambda f: int(m[1]) if (m := re.search(r'_nemo2D_r(\d+)_precrack', f)) else 0)
    out = None
    for f in files:
        p = np.genfromtxt(f, delimiter=',', names=True, ndmin=1)
        if p.size:
            out = p if out is None else np.concatenate([out[out['step'] < p['step'].min()], p])
    return out


def growth_cg(N, P):
    """mean CG per elastic solve from the first cycle the crack grew > 2.5 pixels to the final break"""
    d = load(N, P)
    if d is None:
        return None
    grown = d['crack_length'] > d['crack_length'][0] + 2.5 / N
    i0 = int(np.argmax(grown)) if grown.any() else len(d)
    broken = np.r_[False, d['sigma'][1:] < 0.7 * d['sigma'][:-1]]
    i1 = int(np.argmax(broken)) if broken.any() else len(d)
    return d['cg_u_step'][i0:i1].sum() / d['staggered_its'][i0:i1].sum()


plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.6, 4.6), layout='constrained', gridspec_kw=dict(height_ratios=[1, 0.8]))
for ax in (a1, a2):
    ax.grid(alpha=0.3)
for P, label, c, m in PRECS:
    pts = np.array([(N, v * (2 if P == 'dual' else 1)) for N in GRIDS if (v := growth_cg(N, P)) is not None])
    a1.plot(pts[:, 0], pts[:, 1], '-' + m, color=c, lw=2, ms=5, label=label)
    print(P, dict(zip(pts[:, 0].astype(int), np.round(pts[:, 1]).astype(int))))
a1.axvspan(GRIDS[0] / 1.5, 128, color='0.9', zorder=0)
a1.text(np.sqrt(GRIDS[0] * 128), 330, r'$\ell/h < 2$', ha='center', va='bottom', fontsize=11, color='0.4')
a1.set_xscale('log', base=2)
a1.set_xticks(GRIDS, [f'${n}^2$' for n in GRIDS]), a1.minorticks_off()
a1.set_xlim(GRIDS[0] / 1.15, GRIDS[-1] * 1.15), a1.set_ylim(0, 1000)
a1.set(xlabel='grid size', ylabel='CG iterations per solve')
a1.legend(loc='upper left', fontsize=11, framealpha=0.95)
ends = []
for P, label, c, m in PRECS:
    d = np.genfromtxt(os.path.join(HERE, 'exp_data', f'FAT_N1024_{P}{TAG}_cycles.csv'), delimiter=',', names=True)
    cyc, t = d['step'] + 1, np.cumsum(d['time_u_step'] + d['time_d_step']) / 3600
    a2.plot(cyc, t, '-', color=c, lw=2)
    a2.plot(cyc[-1], t[-1], 'o', color=c, ms=5)
    ends.append((t[-1], cyc[-1], c))
prev = -1e9
for y, x, c in sorted(ends):                              # end labels, kept apart vertically
    y_txt = max(y, prev + 2.0)
    a2.text(x + 5, y_txt, f'{y:.1f} h', color=c, fontsize=11, va='center')
    prev = y_txt
a2.set_xlim(0, 220), a2.set_ylim(0, 15), a2.set_yticks([0, 5, 10, 15])
a2.set(xlabel=r'load cycle ($1024^2$)', ylabel='wall time [h]')
out = os.path.join(HERE, 'figures', 'slide_fatigue_study')
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=150)
    if os.path.isdir(TALK):
        shutil.copy(f'{out}.{ext}', TALK)
print('wrote', out)

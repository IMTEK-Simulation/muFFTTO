"""Fracture-Bench mesh refinement study on NEMO (periodic analogues, fb_dataset1_AT2.py / fb_dataset2_pfczm.py), Green vs
dual (dual CG counted x2). Data: exp_data_nemo/ (rsync of $WS/runs/fracture_bench/exp_data).
  (a) Dataset 2 macro stress vs load step, all grids (Green = dual)
  (b) Dataset 2 mean CG iterations per elastic solve in each load step, Green (solid) vs dual x2 (dashed)
  (c) total elastic CG Green / (2 dual) vs l/h for both datasets, over the load steps completed by BOTH runs
    python plot_fb_mesh_study.py  ->  figures/fb_mesh_study.{png,pdf}
    STEP_MAX=13: Dataset 2 only up to load step 13 (peak + break; later steps = residual-load tail) -> fb_mesh_study_s13"""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
E = os.path.join(HERE, 'exp_data_nemo')
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 12})
rd = lambda f: np.genfromtxt(f, delimiter=',', names=True, ndmin=1)
STEP_MAX = int(os.environ['STEP_MAX']) if 'STEP_MAX' in os.environ else None   # load steps 1 ... STEP_MAX (Dataset 2)
cut = lambda a: a if STEP_MAX is None else a[a['step'] <= STEP_MAX - 1]


def per_step(it):
    """{load step: (sum cg_u, number of elastic solves)}"""
    out = {}
    for s in np.unique(it['step']):
        m = it['step'] == s
        out[int(s)] = (it['cg_u'][m].sum(), m.sum())
    return out


def ratio(g, d):
    """total CG Green / (2 dual) over the load steps finished by both (the last common step included)"""
    common = sorted(set(g) & set(d))
    return sum(g[s][0] for s in common) / (2 * sum(d[s][0] for s in common)), len(common)


fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), layout='constrained')
cols = {128: '#9ecae1', 256: '#3182bd', 512: '#08306b'}
r2 = []
for NX in (128, 256, 512):
    tag = f'FB2_{NX}x{2 * NX}'
    st = {P: cut(rd(glob.glob(os.path.join(E, f'{tag}_{P}_*_steps.csv'))[0])) for P in ('Green', 'dual')}
    it = {P: per_step(cut(rd(glob.glob(os.path.join(E, f'{tag}_{P}_*_iterations.csv'))[0]))) for P in ('Green', 'dual')}
    bh = 0.04 / (4.0 / NX)
    ax[0].plot(st['dual']['step'] + 1, st['dual']['stress'] * 377e3, '-', color=cols[NX], lw=1.8,
               label=rf'${NX} \times {2 * NX}$, $b/h = {bh:.2f}$')
    for P, ls, f in (('Green', '-', 1), ('dual', '--', 2)):
        s = sorted(it[P])
        ax[1].plot(np.array(s) + 1, [f * it[P][k][0] / it[P][k][1] for k in s], ls, color=cols[NX], lw=1.5)
    q, n = ratio(it['Green'], it['dual'])
    r2.append((bh, q))
    print(f'D2 {NX}x{2 * NX}: b/h {bh:.2f}  Green/(2 dual) = {q:.2f} over {n} common load steps')
r1 = []
for N in (128, 256, 512, 1024):
    it = {P: per_step(rd(glob.glob(os.path.join(E, f'AT2_N{N}_{P}_fb_hole*_biax_iterations.csv'))[0])) for P in ('Green', 'dual')}
    q, n = ratio(it['Green'], it['dual'])
    r1.append((0.015 / (2.2 / N), q))
    print(f'D1 {N}^2: l/h {r1[-1][0]:.2f}  Green/(2 dual) = {q:.2f} over {n} load steps')
ttl = '' if STEP_MAX is None else f', load steps 1--{STEP_MAX}'
ax[0].set(xlabel='load step', ylabel='macro stress [MPa]', title='Dataset 2 (PF-CZM, inclusions): stress' + ttl)
ax[0].legend(fontsize=10)
ax[1].plot([], [], 'k-', label='Green'), ax[1].plot([], [], 'k--', label=r'dual ($\times 2$)')
ax[1].set(xlabel='load step', ylabel='CG iterations per solve', title='Dataset 2: elastic CG per solve' + ttl, ylim=(0, None))
ax[1].legend(fontsize=10, loc='upper left')
ax[2].axhline(1, color='0.3', lw=1)
ax[2].axvspan(0.3, 2, color='0.9', zorder=0)
ax[2].plot(*zip(*r1), '-o', color='tab:purple', lw=2, ms=6, label='Dataset 1: AT2, hole + notches (brittle, one-step break)')
ax[2].plot(*zip(*r2), '-s', color='tab:orange', lw=2, ms=6, label='Dataset 2: PF-CZM, inclusions'
           + (' (long cracked tail)' if STEP_MAX is None else f' (load steps 1--{STEP_MAX})'))
ax[2].set_xscale('log', base=2), ax[2].set_yscale('log')
ax[2].set_xticks([0.5, 1, 2, 4, 8], ['0.5', '1', '2', '4', '8']), ax[2].set_yticks([0.25, 0.5, 1, 2, 4], ['0.25', '0.5', '1', '2', '4'])
ax[2].minorticks_off(), ax[2].set_xlim(0.5, 10), ax[2].set_ylim(0.25, 12)   # room above the points for the legend
ax[2].text(9, 1.08, 'dual cheaper', ha='right', va='bottom', fontsize=10), ax[2].text(9, 0.92, 'Green cheaper', ha='right', va='top', fontsize=10)
ax[2].set(xlabel='$l/h$ (D1) or $b/h$ (D2)', ylabel=r'total CG: Green / (dual $\times 2$)', title='Fracture-Bench analogues: who wins')
ax[2].legend(fontsize=9, loc='upper left')
for a in ax:
    a.grid(alpha=0.3)
out = os.path.join(HERE, 'figures', 'fb_mesh_study' + ('' if STEP_MAX is None else f'_s{STEP_MAX}'))
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=150)
print('wrote', out)

"""
Slide figures (LaTeX fonts, slide size, log y 20 - 2000 like the TO slides) for the phase-field fracture block of
Freiburg_GAMM_2026.tex. Mean CG iterations per elasticity solve DURING cracking, dual counted x2 (2D and 3D). Numbers are the study summaries (sources below).
  frac_gj_battery     battery, Green + Green-Jacobi only (slide 'Green and Green-Jacobi')
  frac_battery        battery (Tu 2026) 2D 32^2-1024^2, l = 0.23 um      battery_particle_fracture/plot_battery_mesh.py
  frac_fatigue        fatigue (Carrara 2020) 2D 64^2-512^2, l/h 1-8       phase_field_fatigue/plot_FAT_mesh_study.py
  frac_wu             Wu PF-CZM concrete 2D 32^2-512^2, b/h 0.5-7.7      localization_softening_bands/plot_pfczm_mesh.py
  frac_ch3d           CH two-phase AT2 3D crack step 32^3-256^3, l = 1/16 schneider2025_notch/plot_AT2_ch3D_crack_staggered.py
  frac_grains3d       grains AT2 3D, l/h = 4, 32^3 / 128^3 / 256^3       (NEMO logs, 2026-11-06/07)
  frac_ch2d_lh        CH two-phase AT2 2D, 128^2, l shrunk (l/h 0.25-4)  schneider2025_notch/plot_AT2_ch_small_l_cost.py
  frac_summary        cost ratio Green / (c x dual) vs l/h, all studies   plot_underresolved_summary.py
    python plot_slide_fracture.py   ->  figures/frac_*.{png,pdf} + copy to the talk figures/
"""
import csv
import os
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
COL = {'Green': 'tab:green', 'Green-Jacobi': 'tab:blue', 'dual': 'tab:red'}
MK = {'Green': 'o', 'Green-Jacobi': 's', 'dual': '^'}
DLAB = {'dual': r'dual ($\times 2$)'}

# grid -> (Green, Green-Jacobi, dual) per study
DATA = {
    'battery': ([32, 64, 128, 256, 512, 1024], [109, 233, 270, 322, 336, 357], [64, 128, 208, 377, 645, 1321],
                [56, 53, 51, 52, 52, 58], 2),
    'fatigue': ([64, 128, 256, 512], [104, 167, 283, 500], [94, 156, 276, 502], [93, 84, 70, 58], 2),
    'wu': ([32, 64, 128, 256, 512], [162, 316, 510, 662, 706], [77, 132, 271, 498, 836], [71, 101, 110, 108, 100], 2),
    'ch3d': ([32, 64, 128, 256], [70, 320, 249, 324], [35, 88, 79, 153], [36, 61, 42, 40], 3),
    'grains3d': ([32, 128, 256], [76, 498, np.nan], [57, 97, 279], [25, 95, 87], 3),
}


def save(fig, name):
    for ext in ('png', 'pdf'):
        p = os.path.join(HERE, 'figures', f'{name}.{ext}')
        fig.savefig(p, dpi=150)
        if os.path.isdir(TALK):
            shutil.copy(p, TALK)
    plt.close(fig)


def grid_plot(name, key, precs=('Green', 'Green-Jacobi', 'dual')):
    grids, g, gj, d, dim = DATA[key]
    vals = {'Green': g, 'Green-Jacobi': gj, 'dual': 2 * np.array(d, float)}
    fig, ax = plt.subplots(figsize=(4.4, 3.3), layout='constrained')
    for P in precs:
        y = np.array(vals[P], float)
        m = ~np.isnan(y)
        ax.plot(np.array(grids)[m], y[m], '-' + MK[P], color=COL[P], lw=2.2, ms=6, clip_on=False, zorder=5, label=DLAB.get(P, P))
    ax.set_xscale('log', base=2)
    ax.set_xticks(grids), ax.set_xticklabels([f'${n}^{dim}$' for n in grids]), ax.minorticks_off()
    ax.set_xlim(grids[0], grids[-1])
    ax.set_yscale('log')
    yt = [20, 50, 100, 200, 500, 1000, 2000]
    ax.set_yticks(yt), ax.set_yticklabels([str(v) for v in yt]), ax.minorticks_off()
    ax.set_ylim(20, 2000)
    ax.set(xlabel='grid size', ylabel=r'CG per solve (dual $\times 2$)')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    save(fig, name)


grid_plot('frac_gj_battery', 'battery', ('Green', 'Green-Jacobi'))
for key in ('battery', 'fatigue', 'wu', 'ch3d', 'grains3d'):
    grid_plot(f'frac_{key}', key)

# CH 2D, 128^2, l shrunk below the pixel
lh = [0.25, 0.5, 1, 2, 4]
vals = {'Green': [123, 122, 153, 178, 276], 'Green-Jacobi': [210, 207, 227, 214, 177], 'dual': [204, 190, 150, 126, 124]}       # dual x2
fig, ax = plt.subplots(figsize=(4.4, 3.3), layout='constrained')
for P, y in vals.items():
    ax.plot(lh, y, '-' + MK[P], color=COL[P], lw=2.2, ms=6, clip_on=False, zorder=5, label=DLAB.get(P, P))
ax.set_xscale('log', base=2)
ax.set_xticks(lh), ax.set_xticklabels([f'{v:g}' for v in lh]), ax.minorticks_off()
ax.set_xlim(0.25, 4)
ax.set_yscale('log')
yt = [20, 50, 100, 200, 500, 1000, 2000]
ax.set_yticks(yt), ax.set_yticklabels([str(v) for v in yt]), ax.minorticks_off()
ax.set_ylim(20, 2000)
ax.set(xlabel='$l/h$  ($128^2$, crack resolution)', ylabel=r'CG per solve (dual $\times 2$)')
ax.grid(alpha=0.3)
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
save(fig, 'frac_ch2d_lh')

# summary: cost ratio Green / (c x dual) vs l/h
S = {
    'CH 2D': (2, [(0.25, 123, 102), (0.5, 122, 95), (1, 153, 75), (2, 178, 63), (4, 276, 62)], 'k', 'o'),
    'CH 3D': (3, [(2, 70, 36), (4, 320, 61), (8, 249, 42), (16, 324, 40)], 'k', 's'),
    'fatigue': (2, [(1, 104, 93), (2, 167, 84), (4, 283, 70), (8, 500, 58)], 'tab:purple', 'D'),
    'concrete (PF-CZM)': (2, [(0.48, 162, 71), (0.96, 316, 101), (1.92, 510, 110), (3.84, 662, 108), (7.7, 706, 100)],
                          'tab:orange', 'v'),
    'battery': (2, [(1.23, 109, 56), (2.45, 233, 53), (4.9, 270, 51), (9.8, 322, 52), (19.6, 336, 52), (39, 357, 58)],
                'tab:brown', '^'),
}


def cracked_cg(pat, P):
    """mean elastic CG per solve from the first load step with max d > 0.95 on (AT2 runs; a _restart segment overrides its steps)"""
    root = os.path.join(HERE, 'damage_AT2_small_strain', pat.replace('{P}', P))
    maxd, rows = {}, {}
    for suf in ('', '_restart'):
        log, csvf = root.replace('{d}', 'logs') + suf + '.log', root.replace('{d}', 'exp_data') + suf + '_iterations.csv'
        if not (os.path.exists(log) and os.path.exists(csvf)):
            continue
        maxd.update({int(m[1]): float(m[2]) for m in re.finditer(r'^step\s+(\d+)\s+E_\w+ = \S+\s+sigma_\w+ = \S+\s+max d = (\S+)',
                                                                 open(log).read(), re.M)})
        new = {}
        for r in csv.DictReader(open(csvf)):
            new.setdefault(int(r['step']), []).append(float(r['cg_u']))
        rows.update(new)
    if not rows:
        return None
    crack = min((k for k, v in maxd.items() if v > 0.95), default=None)
    if crack is None:
        return None
    v = [c for k, cs in rows.items() if k >= crack for c in cs]
    return sum(v) / len(v) if v else None


# AT2 series of the talk slides, l = 1/32 fixed (cracked stage, measured from the runs)
for name, tmpl, col, mk, grids in (
        ('porous', '{{d}}/AT2_N{N}_{{P}}_polydisperse_pores_l0.03125', 'tab:cyan', 'P', (32, 64, 128, 256, 1024)),
        ('CH 2D, $l = 1/32$', 'schneider2025_notch/{{d}}/AT2_N{N}_{{P}}_incl0.1_l0.03125_k0.0001_ch', '0.5', 'X', (64, 128, 256, 512))):
    pts = []
    for N in grids:
        g, d = (cracked_cg(tmpl.format(N=N), P) for P in ('Green', 'dual'))
        if g and d:
            pts.append((N / 32, g, d))
    S[name] = (2, pts, col, mk)
    print(name, [(x, round(g), round(d), round(g / (2 * d), 2)) for x, g, d in pts])
# Ernesti 2020 fibres (worktree-polycrystal-crack, ernesti2020_fibers): ratio of total elastic work Green / (2 dual), NOT
# per solve during cracking. 64^2, 128^2: seed 0 (README); 256^2: mean of 8 seeds (README); 512^2: mean of the 7 seeds
# with both runs finished, GPU part after the restart (exp_data/nemo_gpu_512). Stored as (l/h, 2 * ratio, 1).
S['fibres (work)'] = (2, [(0.75, 2 * 0.39, 1), (1.5, 2 * 0.44, 1), (3, 2 * 1.26, 1), (6, 2 * 2.97, 1)], 'tab:pink', '*')
fig, ax = plt.subplots(figsize=(4.4, 3.3), layout='constrained')
for name, (dim, pts, col, mk) in S.items():
    x, g, d = (np.array(v, float) for v in zip(*pts))
    c = 2.0                                                # dual iteration = 2 Green iterations, 2D and 3D
    ax.plot(x, g / (c * d), ('-' if dim == 2 else '--') + mk, color=col, lw=2, ms=5, label=name)
ax.axhline(1, color='0.3', lw=1.2)
ax.axvspan(0.2, 2, color='0.9', zorder=0)
ax.set_xscale('log', base=2), ax.set_yscale('log')
xt = [0.25, 1, 4, 16]
ax.set_xticks(xt), ax.set_xticklabels([f'{v:g}' for v in xt]), ax.minorticks_off()
yt = [0.5, 1, 2, 5]
ax.set_yticks(yt), ax.set_yticklabels([f'{v:g}' for v in yt]), ax.minorticks_off()
ax.set_xlim(0.2, 45), ax.set_ylim(0.4, 7)
ax.set(xlabel='$l/h$', ylabel=r'Green / (dual $\times 2$)')
ax.text(0.27, 0.45, 'Green cheaper', fontsize=10, va='bottom')
ax.text(40, 6.4, 'dual cheaper', fontsize=10, va='top', ha='right')
ax.grid(alpha=0.3)
fig.legend(loc='outside upper center', ncol=3, fontsize=8.5, frameon=False, columnspacing=1.0)
save(fig, 'frac_summary')
print('wrote figures/frac_*.png/.pdf (copied to the talk figures/)')

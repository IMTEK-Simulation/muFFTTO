"""
frac_summary_total with all the other fracture studies that have per-iteration data: total elastic CG over the run,
Green / (2 x dual), vs l/h. Same rules as plot_slide_fracture_whole_run.py (whose SERIES it reuses): compared up to the last
load step reached by BOTH runs (ASR: last common eps_ASR), restarts override. Dual counts x2 in 2D and 3D. Solid = 2D, dashed = 3D.
Not included: grains 3D (only summary numbers, no per-iteration data here), cantilever (other folder and geometry).
    python plot_slide_fracture_whole_run_extended.py  ->  figures/frac_summary_total_extended.{png,pdf} + talk figures/
"""
import csv
import glob
import os
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
src = open(os.path.join(HERE, 'plot_slide_fracture_whole_run.py')).read().split('fig, ax = plt.subplots')[0]
src = src.replace('os.path.realpath(__file__)', repr(os.path.join(HERE, 'plot_slide_fracture_whole_run.py')))
exec(compile(src, 'base', 'exec'))
plt.rcParams.update({'font.size': 13})
ASR = os.path.join(HERE, 'asr_gel_expansion', 'exp_data')
WT = os.path.join(HERE, '..', '..', '.claude', 'worktrees', 'polycrystal-crack', 'experiments', 'dual_preconditioner')


def fatigue3d(N, P):
    D = os.path.join(HERE, 'phase_field_fatigue', 'exp_data')
    fs = [f for f in glob.glob(f'{D}/FAT_N{N}_{P}_incl*_nemo3D*_precrack_*_cycles.csv') if re.search(r'_nemo3D(_r\d+)?_precrack', f)]
    fs.sort(key=lambda f: int(m[1]) if (m := re.search(r'_nemo3D_r(\d+)_precrack', f)) else 0)
    out = {}
    for f in fs:
        out.update(from_steps(f) or {})
    return out or None


def asr_mesh(D, N, P):
    f = f'{ASR}/ASRmesh_{D}D_N{N}_{P}_l0.015625_cheap_iterations.csv'
    if not os.path.exists(f):
        return None
    rows = {(int(r['step']), int(r['staggered_iteration'])): float(r['cg_main']) + float(r['cg_unit'] or 0) for r in csv.DictReader(open(f))}
    out = {}
    for (s, _), v in rows.items():
        out[s] = out.get(s, 0.0) + v
    return out or None


def fourel(N, P):
    base = f'{ASR}/FOUREL_N{N}_{P}_free_k0.0001'
    if not (os.path.exists(base + '_solves.csv') and os.path.exists(base + '_steps.npz')):
        return None
    st = np.load(base + '_steps.npz', allow_pickle=True)
    out = {}
    for r in csv.DictReader(open(base + '_solves.csv')):
        k = round(float(r['eps_asr']), 10)
        out[k] = out.get(k, 0.0) + float(r['linear_its']) + float(r['unit_its'] or 0)
    for e, u in zip(st['eps_asr'], st['unit_its']):
        k = round(float(e), 10)
        out[k] = out.get(k, 0.0) + float(u)
    return out or None


def asr_pf(variant, N, P):
    for d in ('', 'nemo_pf/'):
        f = f'{ASR}/{d}ASRpf_{variant}_N{N}_{P}_steps.npz'
        if os.path.exists(f):
            r = np.load(f, allow_pickle=True)
            return {round(float(e), 10): float(c) for e, c in zip(r['eps_asr'], r['cg_all_sum'])} or None
    return None


def concrete3d(N, P):
    return from_steps(os.path.join(HERE, 'localization_softening_bands', 'exp_data_nemo', f'PFCZM3D_N{N}_{P}_xx_b1.5mm_k1e-05_steps.csv'))


def porous_variant(tag):
    def load(key, P):
        N, l = key
        r = from_iterations(*(f'{AT2}/exp_data/AT2_N{N}_{P}_polydisperse_pores_{tag}l{l}{s}' for s in ('', '_restart')))
        return r or from_iterations(f'{AT2}/exp_data/AT2_N{N}_{P}_polydisperse_pores_{tag}l{l}_lh')     # l/h-study runs
    return load


def cases_porous(tag):
    out = []
    for N in (32, 64, 128, 256, 512, 1024):
        for l in ('0.125', '0.0625', '0.03125', '0.015625', '0.0078125', '0.00390625'):
            if tag == '' and l == '0.03125':
                continue                                             # already the 'porous' series
            out.append(((N, l), N * float(l)))
    return sorted(out, key=lambda c: c[1])


MORE = {
    'battery 3D': (lambda N, P: from_iterations(os.path.join(HERE, 'battery_particle_fracture', 'exp_data', 'nemo', f'battery_N{N}_{P}_ms0.01_3D_tu')),
                   [(N, 0.23 * N / 6) for N in (32, 64, 128, 256)], 'tab:brown', '<', '--'),
    'fatigue 3D': (fatigue3d, [(N, N / 64) for N in (32, 64, 128, 256)], 'tab:purple', 'd', '--'),
    'concrete 3D': (concrete3d, [(N, 1.5 * N / 100) for N in (32, 64, 128, 256)], 'tab:orange', '>', '--'),
    'ASR AT2 2D': (lambda N, P: asr_mesh(2, N, P), [(N, 0.015625 * N) for N in (32, 64, 128, 256, 512, 1024)], 'tab:blue', 'p', '-'),
    'ASR AT2 3D': (lambda N, P: asr_mesh(3, N, P), [(N, 0.015625 * N) for N in (32, 64, 128, 256)], 'tab:blue', 'h', '--'),
    'ASR Fourel nonlocal': (fourel, [(N, N * 0.25 / 35) for N in (64, 128, 256, 512, 1024)], 'tab:pink', 'X', '-'),
    'ASR PF A (AT1)': (lambda N, P: asr_pf('fourel', N, P), [(N, N / 70) for N in (64, 128, 256, 512, 1024)], 'tab:olive', 'P', '-'),
    'ASR PF B (PF-CZM)': (lambda N, P: asr_pf('czm', N, P), [(N, N / 70) for N in (64, 128, 256, 512, 1024)], 'tab:gray', '*', '-'),
    'notch (Schneider 2025)': (lambda N, P: from_iterations(f'{CH}/AT2_N{N}_{P}_l0.02_k0.0001_notch'), [(N, 0.02 * N) for N in (64, 128, 256, 512)], 'tab:green', '1', '-'),
    'polycrystal': (lambda key, P: from_iterations(os.path.join(WT, 'polycrystal_crack', 'exp_data', f'AT2_N{key[0]}_{P}_poly_{key[1]}')),
                    [((64, 'l0.03125'), 2), ((128, 'l0.03125'), 4), ((128, 'gb0.3_l0.015625'), 2)], 'darkred', '2', ''),
    'gypsum, holes': (lambda N, P: from_iterations(f'{AT2}/gypsum_rve/exp_data/AT2_N{N}_{P}_polydisperse_pores_holes0.0001_l0.0195915_gypsum'),
                      [(N, 0.0195915 * N) for N in (64, 128, 256, 512, 1024)], 'teal', '3', ''),
    'gypsum, incl. 10': (lambda N, P: from_iterations(f'{AT2}/gypsum_rve/exp_data/AT2_N{N}_{P}_polydisperse_pores_incl10_l0.0195915_gypsum'),
                         [(N, 0.0195915 * N) for N in (64, 128, 256, 512, 1024)], 'teal', '4', ''),
    'porous, other $l$': (porous_variant(''), cases_porous(''), 'tab:cyan', '.', ''),
    'porous, incl. 10': (porous_variant('incl10_'), cases_porous('incl10_'), 'c', '+', ''),
    'porous, incl. 1000': (porous_variant('incl1000_'), cases_porous('incl1000_'), 'c', 'x', ''),
    'porous, void holes': (porous_variant('holes0.0001_'), cases_porous('holes0.0001_'), 'c', '|', ''),
}
fig, ax = plt.subplots(figsize=(11.5, 7.2), layout='constrained')
for name, (loader, cases, col, mk, ls) in {**SERIES, **MORE}.items():
    pts = []
    for key, lh in cases:
        g, d = loader(key, 'Green'), loader(key, 'dual')
        if not g or not d:
            continue
        last = min(max(g), max(d))
        G, D = (sum(v for s, v in x.items() if s <= last) for x in (g, d))
        pts.append((lh, G / (2 * D)))
        flag = '' if max(g) == max(d) else f'  [cut: Green to {max(g):g}, dual to {max(d):g}]'
        print(f'{name:24s} {str(key):>22s} l/h {lh:6.2f}  ratio {G / (2 * D):5.2f}{flag}')
    if pts:
        ax.plot(*zip(*pts), (ls or '') + mk, color=col, lw=1.8, ms=7, mew=1.3, label=name)
    else:
        print(name, 'no data')
ax.plot([0.75, 1.5, 3, 6], [0.39, 0.44, 1.26, 2.97], '-*', color='magenta', lw=1.8, ms=9, label='fibres')
ax.axhline(1, color='0.3', lw=1.3)
ax.axvline(4, color='0.5', lw=1, ls=':')
ax.axvspan(0.15, 2, color='0.92', zorder=0)
ax.set_xscale('log', base=2), ax.set_yscale('log')
ax.set_xticks([0.25, 0.5, 1, 2, 4, 8, 16, 32]), ax.set_xticklabels(['0.25', '0.5', '1', '2', '4', '8', '16', '32']), ax.minorticks_off()
yt = [0.1, 0.2, 0.5, 1, 2, 5, 10]
ax.set_yticks(yt), ax.set_yticklabels([f'{v:g}' for v in yt]), ax.minorticks_off()
ax.set_xlim(0.15, 45), ax.set_ylim(0.08, 12)
ax.set(xlabel='$l/h$', ylabel=r'total CG: Green / (dual $\times 2$)   ($>1$: dual cheaper)')
ax.grid(alpha=0.3)
fig.legend(loc='outside right upper', fontsize=11, frameon=False)
for ext in ('png', 'pdf'):
    p = os.path.join(HERE, 'figures', f'frac_summary_total_extended.{ext}')
    fig.savefig(p, dpi=170)
    shutil.copy(p, TALK)
print('wrote', p)

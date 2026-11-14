"""
Whole-run version of frac_summary (plot_slide_fracture.py): TOTAL elastic CG iterations over the whole run, Green / (2 x
dual), against l/h, from the per-iteration CSVs. No window: pre-crack, cracking and post-peak all count. Restart segments
override the steps they repeat. Green and dual are compared up to the last load step reached by BOTH runs (printed).
Fibres: total work from the README (64-256^2) and the GPU part after the restart at 512^2 (see plot_slide_fracture.py).
    python plot_slide_fracture_whole_run.py  ->  figures/frac_summary_total.{png,pdf} + talk figures/
    SHORT=1: 6.0 x 1.45 in for the combined slide with the crack-resolution row (fibre_crack_row_lh256), y label without
        'total CG', x ticks at the crack columns' l/h (0.75, 2, 4) and 16  ->  figures/frac_summary_total_short.{png,pdf}
"""
import csv
import glob
import os
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
AT2 = os.path.join(HERE, 'damage_AT2_small_strain')
CH = os.path.join(AT2, 'schneider2025_notch', 'exp_data')


def from_iterations(*stems):
    """{step: total cg_u}; later files (restarts) override the (step, staggered iteration) they repeat"""
    rows = {}
    for stem in stems:
        f = stem + '_iterations.csv'
        if os.path.exists(f):
            for r in csv.DictReader(open(f)):
                rows[(int(r['step']), int(r['staggered_iteration']))] = float(r['cg_u'])
    out = {}
    for (s, _), v in rows.items():
        out[s] = out.get(s, 0.0) + v
    return out or None


def from_steps(f):
    if not os.path.exists(f):
        return None
    return {int(r['step']): float(r['cg_u_step']) for r in csv.DictReader(open(f))} or None


def fatigue(N, P):
    D = os.path.join(HERE, 'phase_field_fatigue', 'exp_data')
    fs = [f for f in glob.glob(f'{D}/FAT_N{N}_{P}_incl*_nemo2D*_precrack_*_cycles.csv') if re.search(r'_nemo2D(_r\d+)?_precrack', f)]
    fs.sort(key=lambda f: int(m[1]) if (m := re.search(r'_nemo2D_r(\d+)_precrack', f)) else 0)
    out = {}
    for f in fs:
        out.update(from_steps(f) or {})
    return out or None


def concrete(N, P):
    for d in ('exp_data_nemo', 'exp_data'):
        f = os.path.join(HERE, 'localization_softening_bands', d, f'PFCZM2D_N{N}_{P}_xx_b1.5mm_k1e-05_steps.csv')
        if os.path.exists(f):
            return from_steps(f)
    return None


SERIES = {
    'Cahn-Hilliard 2D': (lambda x, P: from_iterations(f'{CH}/AT2_N128_{P}_incl0.1_l{x}_k0.0001_ch'),
              [('0.00195312', 0.25), ('0.00390625', 0.5), ('0.0078125', 1), ('0.015625', 2), ('0.03125', 4)], 'k', 'o', '-'),
    'Cahn-Hilliard 3D': (lambda x, P: from_iterations(f'{CH}/nemo_3D/AT2_N{x}_{P}_incl0.1_l0.0625_3D_k0.0001_ch'),
              [(32, 2), (64, 4), (128, 8), (256, 16)], 'k', 's', '--'),
    'fatigue (Carrara 2020)': (fatigue, [(N, N / 64) for N in (64, 128, 256, 512, 1024)], 'tab:purple', 'D', '-'),
    # PF-CZM length b = 1.5 mm in a 100 mm cell. Its 1D profile 1 - sin(|x|/b) is narrower than AT2's exp(-|x|/l):
    # same crack width at l_AT2 ~ 0.5 b (d > 0.1 band: 0.49 b, integral of d: 0.57 b), so plotted at l/h = 0.5 b/h
    'cohesive PF fracture (Wu 2017)': (concrete, [(N, 0.5 * 1.5 * N / 100) for N in (32, 64, 128, 256, 512)], 'tab:orange', 'v', '-'),
    # 6 um cell, l = 0.23 um (core) / 0.27 um (shell, where the crack starts): plotted with the shell l
    'Li-ion battery particle (Tu 2026)': (lambda N, P: from_iterations(os.path.join(HERE, 'battery_particle_fracture', 'exp_data', 'nemo', f'battery_N{N}_{P}_ms0.01_tu')),
                [(N, 0.27 * N / 6) for N in (32, 64, 128, 256, 512, 1024)], 'tab:brown', '^', '-'),
    'porous': (lambda N, P: from_iterations(*(f'{AT2}/exp_data/AT2_N{N}_{P}_polydisperse_pores_l0.03125{s}' for s in ('', '_restart'))),
               [(N, N / 32) for N in (32, 64, 128, 256, 1024)], 'tab:cyan', 'P', '-'),
    'Cahn-Hilliard 2D, $l = 1/32$': (lambda N, P: from_iterations(*(f'{CH}/AT2_N{N}_{P}_incl0.1_l0.03125_k0.0001_ch{s}' for s in ('', '_restart'))),
                          [(N, N / 32) for N in (64, 128, 256, 512)], '0.5', 'X', '-'),
}
SHORT = os.environ.get('SHORT') == '1'
if SHORT:
    plt.rcParams.update({'font.size': 10.5})
fig, ax = plt.subplots(figsize=(6.0, 1.45 if SHORT else 2.85), layout='constrained')   # slide: width = \textwidth
for name, (loader, cases, col, mk, ls) in SERIES.items():
    pts = []
    for key, lh in cases:
        g, d = loader(key, 'Green'), loader(key, 'dual')
        if not g or not d:
            print(f'{name}: {key} missing', 'Green' if not g else 'dual')
            continue
        last = min(max(g), max(d))
        G, D = (sum(v for s, v in x.items() if s <= last) for x in (g, d))
        pts.append((lh, G / (2 * D)))
        print(f'{name:20s} key {str(key):>11s} l/h {lh:6.2f}  Green {G:12.0f}  dual x2 {2 * D:12.0f}  ratio {G / (2 * D):5.2f}  '
              f'last common step {last} (Green {max(g)}, dual {max(d)})')
    ax.plot(*zip(*pts), ls + mk, color=col, lw=2, ms=5, label=name)
fib = [(0.75, 0.39), (1.5, 0.44), (3, 1.26), (6, 2.97)]
ax.plot(*zip(*fib), '-*', color='tab:pink', lw=2, ms=7, label='fibre composite (Ernesti 2020)')
# fibres, fixed 256^2 grid, only l changed: total elastic CG over the whole run, Green / (dual x2), per fibre seed.
# seed 0: worktree ernesti2020_fibers; seeds 1-3: NEMO runs with suffix _lhstudy (2026-11-10), ernesti2020_fibers_lhstudy/exp_data.
fib_seeds = {0.75: [0.58, 0.55, 0.55, 0.42],1.5: [0.85, 0.67, 0.79, 0.53], 2: [0.78, 0.70, 0.86, 0.71], 3: [1.25, 1.00, 1.23, 0.84],
             4: [1.17, 1.33, 2.17, 2.73], 6: [1.92, 1.55, 2.83, 3.50]}
lhs = sorted(fib_seeds)
band = [lh for lh in lhs if len(fib_seeds[lh]) > 1]
ax.fill_between(band, [min(fib_seeds[lh]) for lh in band], [max(fib_seeds[lh]) for lh in band], color='tab:olive', alpha=0.25, lw=0)
ax.plot(lhs, [np.mean(fib_seeds[lh]) for lh in lhs], '-h', color='tab:olive', lw=2, ms=5,
        label='fibre composite, $256^2$, $l$ varied (4 seeds)')
ax.axhline(1, color='0.3', lw=1.2)
ax.axvspan(0.2, 2, color='0.9', zorder=0)
ax.set_xscale('log', base=2), ax.set_yscale('log')
XT = [0.75, 2, 4, 16] if SHORT else [0.25, 1, 4, 16]
ax.set_xticks(XT), ax.set_xticklabels([f'{v:g}' for v in XT]), ax.minorticks_off()
yt = [0.5, 1, 2, 5]
ax.set_yticks(yt), ax.set_yticklabels([f'{v:g}' for v in yt]), ax.minorticks_off()
ax.set_xlim(0.2, 45), ax.set_ylim(0.2, 7)
ax.set(xlabel='$l/h$', ylabel='Green / dual' if SHORT else r'total CG: Green / (dual $\times 2$)')
if SHORT:
    ax.xaxis.labelpad = 0
ax.text(0.27, 0.23, 'Green cheaper', fontsize=9 if SHORT else 10, va='bottom')
ax.text(40, *((0.92, 'dual cheaper') if SHORT else (1.08, 'dual cheaper')), fontsize=9 if SHORT else 10, va='top' if SHORT else 'bottom', ha='right')  # SHORT: below the ratio-1 line   # just above ratio 1, empty corner
ax.grid(alpha=0.3)
H, L = ax.get_legend_handles_labels()
if SHORT:   # one legend row for the 2D and 3D Cahn-Hilliard curves (both drawn: solid 2D, dashed 3D)
    from matplotlib.legend_handler import HandlerTuple
    i2, i3 = L.index('Cahn-Hilliard 2D'), L.index('Cahn-Hilliard 3D')
    H[i2], L[i2] = (H[i2], H[i3]), 'Cahn-Hilliard 2D / 3D'
    del H[i3], L[i3]
    i = L.index('Cahn-Hilliard 2D, $l = 1/32$')   # curve kept in the plot (grey), no legend row on the combined slide
    del H[i], L[i]
fig.legend(H, L, loc='outside right upper' if SHORT else 'outside right center', ncol=1, fontsize=8 if SHORT else 10.5, frameon=False, handlelength=2.8 if SHORT else 1.8,
           title='problem type', title_fontsize=8.5 if SHORT else 11, alignment='left',
           **(dict(labelspacing=0.12, borderpad=0.1, handletextpad=0.5, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)}) if SHORT else {}))
for ext in ('png', 'pdf'):
    p = os.path.join(HERE, 'figures', f'frac_summary_total{"_short" if SHORT else ""}.{ext}')
    fig.savefig(p, dpi=200)
    shutil.copy(p, TALK)
if SHORT:   # tick-label anchors (figure fractions, bottom centre of 0.75, 2, 4) for the TikZ connector lines on the slide
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    lab = {t.get_text(): t.get_window_extent(r) for t in ax.get_xticklabels()}
    lines = [f'\\def\\fracTickY{{{min(lab[k].y0 for k in ("0.75", "2", "4")) / fig.bbox.height:.4f}}}']
    lines += [f'\\def\\fracTickX{n}{{{(lab[k].x0 + lab[k].x1) / 2 / fig.bbox.width:.4f}}}' for n, k in zip('abc', ('0.75', '2', '4'))]
    lines += [f'\\def\\fracTopAspect{{{fig.bbox.height / fig.bbox.width:.4f}}}']
    xl = ax.xaxis.label.get_window_extent(r)   # centred l/h label: the connector lines pass beside / below it
    lines += [f'\\def\\fracLabelL{{{xl.x0 / fig.bbox.width:.4f}}}', f'\\def\\fracLabelR{{{xl.x1 / fig.bbox.width:.4f}}}',
              f'\\def\\fracLabelB{{{xl.y0 / fig.bbox.height:.4f}}}']
    t = os.path.join(HERE, 'figures', 'frac_summary_total_short_ticks.tex')
    open(t, 'w').write('% written by plot_slide_fracture_whole_run.py (SHORT=1)\n' + '\n'.join(lines) + '\n')
    shutil.copy(t, TALK)
    print(open(t).read())
print('wrote', p)

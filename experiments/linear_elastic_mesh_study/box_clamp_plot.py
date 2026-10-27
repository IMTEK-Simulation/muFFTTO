"""
Figure and table of run_box.sh (box_clamp_study.py): corrected Green with the fixed dofs on full grid lines.

    1. 2 parallel lines: total Green applications (setup + solve) vs N, dense S / inner CG / exact FFT
    2. 4 faces: the same, dense S / inner CG / inner PCG with the family FFT blocks
    3. 4 faces: inner steps per outer iteration
    the table has every variant (also the per-line FFT, other tolerances, the family FFT applied once)

usage:  python box_clamp_plot.py      -> exp_data/box_clamp_study.png, exp_data/box_clamp_study.md
"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.realpath(__file__))
RUNS = os.path.join(HERE, 'exp_data', 'runs_box')
NS = (32, 64, 128, 256)
# (clamp, variant, label, colour, line style, marker)
SERIES = [('two', 'dense', '2 parallel lines: dense S', '0.25', '-', 'v'),
          ('two', 'familydirect', '2 parallel lines: FFT solve, no iteration', 'tab:green', '-', 'D'),
          ('two', 'cg1e-08', '2 parallel lines: inner CG 1e-8', 'tab:olive', ':', 'o'),
          ('box', 'dense', '4 faces: dense S', 'k', '-', 'v'),
          ('box', 'cg1e-08', '4 faces: inner CG 1e-8', 'tab:blue', ':', 'o'),
          ('box', 'line1e-06', '4 faces: inner PCG, per-line FFT 1e-6', 'tab:purple', '--', 's'),
          ('box', 'family1e-04', '4 faces: inner PCG, family FFT 1e-4', '#fdae6b', '--', '^'),
          ('box', 'family1e-06', '4 faces: inner PCG, family FFT 1e-6', '#f16913', '--', '^'),
          ('box', 'family1e-08', '4 faces: inner PCG, family FFT 1e-8', '#a63603', '--', '^'),
          ('box', 'familydirect', '4 faces: family FFT once, no iteration', 'tab:red', '-.', 'D')]


def load(n, clamp, variant):
    f = os.path.join(RUNS, f'N{n}_{clamp}_{variant}.npz')
    return {k: v[()] for k, v in np.load(f).items()} if os.path.exists(f) else None


rows = []
for clamp, variant, label, colour, ls, marker in SERIES:
    for n in NS:
        r = load(n, clamp, variant)
        if r is None:
            continue
        ref = load(n, clamp, 'dense')
        err = abs(float(r['compliance']) / float(ref['compliance']) - 1) if ref else np.nan
        rows.append((clamp, n, int(r['m']), label.split(': ')[1], int(r['cg_its']), float(r['inner_its'].mean())
                     if len(r['inner_its']) else 0.0, int(r['setup_apps']), int(r['green_apps']), float(r['time']), err))

# a few series per panel, each labelled at its last point; colour = how S is solved, the same in every panel
plt.rcParams.update({'font.size': 12})
DENSE, PLAIN, FFT, ONCE = 'k', 'tab:blue', 'tab:orange', 'tab:red'


def series(ax, clamp, variant, value, colour, text, ls='-', marker='o', below=False, end_dy=-4, numbers=False):
    pts = [(n, load(n, clamp, variant)) for n in NS]
    pts = [(n, value(r)) for n, r in pts if r is not None]
    ax.plot([n for n, _ in pts], [v for _, v in pts], ls=ls, marker=marker, color=colour, lw=2, ms=7)
    for n, v in (pts if numbers else []):
        ax.annotate(f'{v:.0f}', (n, v), textcoords='offset points', xytext=(0, -16 if below else 7), ha='center', fontsize=9,
                    color=colour)
    ax.annotate(f'{text}: {pts[-1][1]:.0f}', (pts[-1][0], pts[-1][1]), textcoords='offset points', xytext=(12, end_dy), fontsize=11,
                color=colour)


total = lambda r: int(r['green_apps']) + int(r['setup_apps'])
fig, axes = plt.subplots(1, 3, figsize=(19, 6), layout='constrained')
ax = axes[0]
series(ax, 'two', 'dense', total, DENSE, 'dense S', end_dy=6)
series(ax, 'two', 'cg1e-08', total, PLAIN, 'inner CG', below=True, end_dy=-14)
series(ax, 'two', 'familydirect', total, FFT, 'FFT, exact\n(one step)')
ax.set_title('2 parallel clamped lines\ntotal cost: Green applications', fontsize=13)
ax = axes[1]
series(ax, 'box', 'dense', total, DENSE, 'dense S')
series(ax, 'box', 'cg1e-08', total, PLAIN, 'inner CG', below=True)
series(ax, 'box', 'family1e-06', total, FFT, 'inner PCG\nwith FFT blocks')
ax.set_title('all 4 faces clamped\ntotal cost: Green applications', fontsize=13)
for ax in axes[:2]:
    ax.set(yscale='log', ylim=(40, 8000), ylabel='Green applications (setup + solve)')
ax = axes[2]
series(ax, 'box', 'cg1e-08', lambda r: r['inner_its'].mean(), PLAIN, 'inner CG')
series(ax, 'box', 'family1e-06', lambda r: r['inner_its'].mean(), FFT, 'inner PCG\nwith FFT blocks')
ax.set(ylim=(0, 90), ylabel='inner steps per outer iteration')
ax.set_title('all 4 faces clamped\ninner steps per outer CG iteration', fontsize=13)
for ax in axes:
    ax.set_xscale('log')
    ax.set_xticks(NS, [str(n) for n in NS])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(26, 600)
    ax.set_xlabel('N (pixels over the square)')
    ax.grid(True, color='0.92', which='both')
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Corrected Green, fixed dofs on full grid lines. Every variant shown gives the same outer CG count as the '
             'dense S\n(2 lines: 31, 4 faces: 28-30) and the same compliance (to 1e-11)', fontsize=13)
fig.savefig(os.path.join(HERE, 'exp_data', 'box_clamp_study.png'), dpi=110)
with open(os.path.join(HERE, 'exp_data', 'box_clamp_study.md'), 'w') as out:
    out.write('| clamp | N | m | S solve | outer its | inner/outer | setup Green apps | solve Green apps | time [s] | '
              'compliance err |\n|' + '---|' * 10 + '\n')
    for r in sorted(rows, key=lambda r: (r[0], r[1])):
        out.write(f'| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]:.1f} | {r[6]} | {r[7]} | {r[8]:.1f} | '
                  f'{r[9]:.1e} |\n')
print('wrote exp_data/box_clamp_study.png, exp_data/box_clamp_study.md')

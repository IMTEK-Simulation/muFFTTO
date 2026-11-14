"""Three-slide sequence, phase-field TO, CG iterations per linear solve against the grid (contrast_summary.csv), one figure
per preconditioner (Green, Green-Jacobi), measured points only (Green was not run at 1024^2):
  1. l = 2h, contrasts 10^2 - 10^5, light -> dark shades, solid lines, circles    -> slide_{green,gj}_contrasts_l2h
  2. l = 2h, contrast 10^4 only                                                    -> slide_{green,gj}_k1e4_l2h
  3. contrast 10^4, l = 1h, 2h, 4h, 8h, one shade, line style and marker            -> slide_{green,gj}_k1e4_all_l
All markers filled (also where the design ended grey on coarse grids).
    python plot_slide_TO_contrast_l_story.py  ->  figures/ (+ copy to the talk figures/)
"""
import csv
import os
import shutil

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
GRIDS = [32, 64, 128, 256, 512, 1024]
D = {}
for r in csv.DictReader(open(os.path.join(HERE, 'figures', 'nemo', 'contrast_summary.csv'))):
    D[(r['precond'], int(float(r['contrast'])), int(float(r['eta_h'])), int(r['N']))] = (float(r['cg']), r['design'] == 'True')
PANELS = [('Green', 'green', '#228b22', ['#9be29b', '#4fbf4f', '#228b22', '#0b4d0b']),
          ('Green_Jacobi', 'gj', '#2171b5', ['#9ecae1', '#4292c6', '#2171b5', '#08306b'])]
STY = [('-', 'o'), ('--', 's'), ('-.', '^'), (':', 'D')]
K = {100: '10^{-2}', 1000: '10^{-3}', 10000: '10^{-4}', 100000: '10^{-5}'}
VOID = r'\kappa_{\mathrm{void}}'
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})


def figure(P, series, name, ncol=2, direct=False, inside=False, loc='upper left', reverse=False):
    """series: list of (contrast, eta_h, colour, line style, marker, label)"""
    fig = plt.figure(figsize=(4.4, 3.7))
    ax = fig.add_axes([0.16, 0.16, 0.76, 0.80 if inside else 0.63])  # same plot box; inside: no legend above, taller plot
    for kap, e, col, ls, mk, lab in series:
        pts = [(n, *D[(P, kap, e, n)]) for n in GRIDS if (P, kap, e, n) in D]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=col, lw=2.2, clip_on=False)
        for n, v, ok in pts:
            ax.plot(n, v, mk, color=col, ms=6, mew=1.6, clip_on=False, zorder=5)
        ax.plot([], [], ls + mk, color=col, lw=2.2, ms=6, label=lab)
        print(name, lab, [round(p[1]) for p in pts])
        if direct:                                               # label on the line (as the original Green plot)
            ax.text(430, (pts[-1][1] * pts[-2][1]) ** 0.5 / 1.22, lab, color=col, ha='center', va='top', fontsize=12)
    ax.set_xscale('log', base=2)
    ax.set_xticks(GRIDS), ax.set_xticklabels([f'{n}$^2$' for n in GRIDS]), ax.minorticks_off()
    ax.set_xlim(32, 1024)
    ax.set_yscale('log')
    yt = [20, 50, 100, 200, 500, 1000, 2000]
    ax.set_yticks(yt), ax.set_yticklabels([str(v) for v in yt]), ax.minorticks_off()
    ax.set_ylim(20, 2000)
    ax.set(xlabel='grid size', ylabel='CG iterations per solve')
    ax.grid(alpha=0.3)
    if inside and not direct:
        ax.legend(loc=loc, fontsize=12, handlelength=2.6, framealpha=0.95, reverse=reverse)  # reverse: highest contrast on top
    elif not direct:
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=ncol, fontsize=12, columnspacing=1.2,
                  handlelength=2.6, frameon=False)
    out = os.path.join(HERE, 'figures', name)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', dpi=200)
        shutil.copy(f'{out}.{ext}', TALK)
    plt.close(fig)


for P, tag, col, shades in PANELS:
    figure(P, [(k, 2, shades[i], '-', 'o', f'${VOID} = {K[k]}$') for i, k in enumerate(K)], f'slide_{tag}_contrasts_l2h',
           direct=P == 'Green', inside=True, reverse=True)
    figure(P, [(10000, 2, col, '-', 'o', f'${VOID} = 10^{{-4}}$')], f'slide_{tag}_k1e4_l2h', inside=True)
    figure(P, [(10000, e, col, *STY[i], f'$l = {e}h$') for i, e in enumerate((1, 2, 4, 8))],
           f'slide_{tag}_k1e4_all_l', inside=True, loc='lower right' if P == 'Green' else 'upper left')

"""
Topology optimisation counterpart of ../plot_underresolved_summary.py: does an under-resolved interface favour Green?
Phase-field TO (eta = interface length), NEMO eta / contrast study (figures/nemo/contrast_summary.csv: N = 32 ... 1024,
eta/h = 1, 2, 4, 8, contrast 1e2 ... 1e5; 'work' = CG work per linear solve over the run, dual counted x 2).
  (a) Green / dual work against eta/h, one line per grid (contrast 1e4)
  (b) the same against the grid N, one line per eta/h (contrast 1e4)
  (c) Green / dual against eta/h for every contrast (512^2)
  (d) Green-Jacobi / dual against eta/h, one line per grid (contrast 1e4)
> 1: dual cheaper, < 1: the other one cheaper.  Green was not run at 1024^2.
    python plot_TO_resolution_green_vs_dual.py   ->  figures/TO_resolution_green_vs_dual.png
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
W = {}
for r in csv.DictReader(open(os.path.join(HERE, 'figures', 'nemo', 'contrast_summary.csv'))):
    W[(int(float(r['contrast'])), int(r['N']), int(float(r['eta_h'])), r['precond'])] = float(r['work'])
GRIDS = [32, 64, 128, 256, 512, 1024]
EH = [1, 2, 4, 8]
KAPPA = [100, 1000, 10000, 100000]
plt.rcParams.update({'font.size': 13})
cN = plt.cm.viridis(np.linspace(0, 0.9, len(GRIDS)))
cE = plt.cm.plasma(np.linspace(0, 0.85, len(EH)))
cK = plt.cm.cividis(np.linspace(0, 0.9, len(KAPPA)))


def ratio(k, n, e, other='Green'):
    a, b = W.get((k, n, e, other)), W.get((k, n, e, 'dual'))
    return a / b if a and b else np.nan


fig, axs = plt.subplots(2, 2, figsize=(16, 12), layout='constrained')
(a, b), (c, d) = axs
for n, col in zip(GRIDS, cN):
    a.plot(EH, [ratio(10000, n, e) for e in EH], 'o-', color=col, lw=2.2, ms=8, label=f'{n}$^2$')
    d.plot(EH, [ratio(10000, n, e, 'Green_Jacobi') for e in EH], 's-', color=col, lw=2.2, ms=8, label=f'{n}$^2$')
for e, col in zip(EH, cE):
    b.plot(GRIDS, [ratio(10000, n, e) for n in GRIDS], 'o-', color=col, lw=2.2, ms=8, label=f'eta = {e} h')
for k, col in zip(KAPPA, cK):
    c.plot(EH, [ratio(k, 512, e) for e in EH], 'o-', color=col, lw=2.2, ms=8, label=f'contrast {k:.0e}')

for ax, t, yl in [(a, '(a) Green / dual vs eta/h, contrast 1e4', 'Green work / dual work (x 2)'),
                  (b, '(b) Green / dual vs grid, contrast 1e4', 'Green work / dual work (x 2)'),
                  (c, '(c) Green / dual vs eta/h, 512$^2$, all contrasts', 'Green work / dual work (x 2)'),
                  (d, '(d) Green-Jacobi / dual vs eta/h, contrast 1e4', 'Green-Jacobi work / dual work (x 2)')]:
    ax.set_xscale('log', base=2), ax.set_yscale('log')
    ax.axhline(1, color='k', lw=1.5)
    yt = [0.2, 0.5, 1, 2, 5, 10]
    ax.set_yticks(yt), ax.set_yticklabels([f'{v:g}' for v in yt])
    ax.set_ylim(0.15, 12)
    ax.set(title=t, ylabel=yl)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
for ax in (a, c, d):
    ax.set_xticks(EH), ax.set_xticklabels([str(e) for e in EH]), ax.minorticks_off()
    ax.axvspan(0.8, 2, color='0.9', zorder=0)
    ax.set_xlim(0.8, 10)
    ax.set_xlabel('eta / h  (interface length per pixel; grey: < 2 under-resolved)')
b.set_xticks(GRIDS), b.set_xticklabels([str(n) for n in GRIDS]), b.minorticks_off()
b.set_xlabel('grid N (N$^2$)')
a.text(0.9, 0.2, 'above 1: dual cheaper', fontsize=11)
fig.suptitle('Phase-field topology optimisation (NEMO eta / contrast study): dual vs Green against the interface '
             'resolution', fontsize=15)
out = os.path.join(HERE, 'figures', 'TO_resolution_green_vs_dual.png')
fig.savefig(out, dpi=120)
for k in KAPPA:
    for n in GRIDS:
        print(k, n, ' '.join(f'{ratio(k, n, e):5.2f}' for e in EH), ' GJ:',
              ' '.join(f'{ratio(k, n, e, "Green_Jacobi"):5.2f}' for e in EH))
print('wrote', out)

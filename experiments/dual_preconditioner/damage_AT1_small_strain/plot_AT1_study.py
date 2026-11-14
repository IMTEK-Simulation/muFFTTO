"""AT1 NEMO study (B2 nine pores), iterations and results only:
  (a) fixed l = 1/32: elastic CG per solve vs grid (Green, Green-Jacobi, dual x2)
  (b) fixed l/h (l shrinks with the grid): dual x2 (solid) and Green (dashed) vs grid
  (c) peak sigma_xx vs grid, every series (fixed l converges, fixed l/h grows as l^(-1/2))
  (d) sigma-E curves of the l = 1/32 series
Input: exp_data/nemo/study/AT1_study_summary.csv (summarize_AT1_study.py) and logs/nemo_study/*.log
    python plot_AT1_study.py  ->  figures/AT1_study.png"""
import os, re, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__))
R = list(csv.DictReader(open(os.path.join(H, 'exp_data/nemo/study/AT1_study_summary.csv'))))
COL = {'Green': '#1baf7a', 'Green_Jacobi': '#2a78d6', 'dual': '#d6402a'}
LAB = {'Green': 'Green', 'Green_Jacobi': 'Green-Jacobi', 'dual': r'dual $\times 2$'}
NS = [32, 64, 128, 256, 512, 1024]
LH = [8, 4, 2, 1, 0.5, 0.25]
lhcol = dict(zip(LH, plt.cm.viridis(np.linspace(0.0, 0.9, len(LH)))))


def get(N, l, P, key):
    x = [float(r[key]) for r in R if int(r['N']) == N and abs(float(r['l']) - l) < 1e-4 * l and r['P'] == P]
    return x[0] if x else None


def series(lf, P, key):
    pts = [(N, get(N, lf(N), P, key)) for N in NS]
    pts = [(N, v * (2 if key == 'cg_u' and P == 'dual' else 1)) for N, v in pts if v is not None]
    return np.array(pts).T if pts else None


fig, ax = plt.subplots(2, 2, figsize=(12, 8.5), layout='constrained')
a = ax[0, 0]
for P in COL:
    x, y = series(lambda N: 1 / 32, P, 'cg_u')
    a.plot(x, y, 'o-', color=COL[P], lw=2, label=LAB[P])
    a.annotate(f'{y[-1]:.0f}', (x[-1], y[-1]), xytext=(6, 0), textcoords='offset points', va='center', color=COL[P])
a.set(title=r'(a) fixed $l = 1/32$: elastic CG per solve', ylabel='CG iterations per solve', ylim=(0, None))
a.legend(frameon=False)

a = ax[0, 1]
for lh in LH:
    for P, ls in (('dual', '-'), ('Green', '--')):
        s = series(lambda N: lh / N, P, 'cg_u')
        a.plot(*s, ls, marker='o', ms=4, color=lhcol[lh], lw=1.8, label=f'$l/h = {lh:g}$' if P == 'dual' else None)
a.plot([], [], 'k-', label=r'dual $\times 2$'); a.plot([], [], 'k--', label='Green')
a.set(title=r'(b) fixed $l/h$ ($l$ shrinks with the grid)', ylabel='CG iterations per solve', ylim=(0, None))
a.legend(frameon=False, fontsize=9, ncol=2)

a = ax[1, 0]
x, y = series(lambda N: 1 / 32, 'dual', 'peak')
a.plot(x, y * 1e2, 'ks-', lw=2.4, ms=6, label=r'fixed $l = 1/32$')
for lh in LH:
    s = series(lambda N: lh / N, 'dual', 'peak')
    a.plot(s[0], s[1] * 1e2, 'o-', color=lhcol[lh], lw=1.6, ms=4, label=f'$l/h = {lh:g}$')
xr = np.array([128, 1024])
a.plot(xr, 4.6 * (xr / 128) ** 0.5, ':', color='0.4', lw=1.2, label=r'$\propto l^{-1/2}$')
a.set_yscale('log'); a.set_yticks([1.5, 2, 3, 5, 10], ['1.5', '2', '3', '5', '10']); a.minorticks_off()
a.set(title=r'(b) peak stress (identical for all preconditioners)'.replace('(b)', '(c)'),
      ylabel=r'peak $\bar\sigma_{xx}$  [$10^{-2}$]')
a.legend(frameon=False, fontsize=9, ncol=2)

a = ax[1, 1]
step_re = re.compile(r'^step\s+\d+\s+E_\w+ = ([0-9.e+-]+)\s+sigma_\w+ = ([0-9.e+-]+)', re.M)
for N, c in zip(NS, plt.cm.Greys(np.linspace(0.35, 1.0, len(NS)))):
    txt = open(os.path.join(H, f'logs/nemo_study/AT1_N{N}_dual_polydisperse_pores_l0.03125.log')).read()
    E, s = np.array(step_re.findall(txt), float).T
    a.plot(np.r_[0, E], np.r_[0, s] * 1e2, color=c, lw=1.8, label=f'${N}^2$')
a.set(title=r'(d) fixed $l = 1/32$: stress-strain', xlabel=r'macroscopic strain $\bar\varepsilon_{xx}$',
      ylabel=r'$\bar\sigma_{xx}$  [$10^{-2}$]', xlim=(0, None), ylim=(0, None))
a.legend(frameon=False, fontsize=9)
for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
    a.set_xscale('log', base=2); a.set_xticks(NS, [f'${n}^2$' for n in NS]); a.set_xlabel('grid')
    a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
for a in ax.flat:
    a.grid(alpha=0.3); a.spines[['top', 'right']].set_visible(False)
fig.suptitle('AT1, B2 nine pores: NEMO study (elastic CG per solve: dual counted $\\times 2$)', fontsize=13)
out = os.path.join(H, 'figures/AT1_study.png'); fig.savefig(out, dpi=130); print(out)

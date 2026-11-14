"""AT1 NEMO study (B2 nine pores): total elastic CG over the whole run, Green / (2 x dual), against l/h (nominal, AT1 l).
One curve per grid (l varied) + the fixed l = 1/32 refinement series. Green and dual are compared up to the last load
step reached by both (as in ../plot_slide_fracture_whole_run.py).
    python plot_AT1_ratio_lh.py  ->  figures/AT1_ratio_lh.png + table in the terminal"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(H, 'exp_data/nemo/study')
NS = [32, 64, 128, 256, 512, 1024]
LH = [0.25, 0.5, 1, 2, 4, 8, 16, 32]


def total(N, P, l):
    f = f'{D}/AT1_N{N}_{P}_polydisperse_pores_l{l:g}_iterations.csv'
    if not os.path.exists(f):
        return None
    out = {}
    for r in csv.DictReader(open(f)):
        out[int(r['step'])] = out.get(int(r['step']), 0.0) + float(r['cg_u'])
    return out


def ratio(N, l):
    g, d = total(N, 'Green', l), total(N, 'dual', l)
    if not g or not d:
        return None
    last = min(max(g), max(d))
    return sum(v for s, v in g.items() if s <= last) / (2 * sum(v for s, v in d.items() if s <= last))


fig, ax = plt.subplots(figsize=(7.5, 4.6), layout='constrained')
cols = plt.cm.viridis(np.linspace(0.0, 0.9, len(NS)))
print('Green / (2 dual), total elastic CG;  rows: grid, columns: l/h =', LH)
for N, c in zip(NS, cols):
    pts = [(lh, ratio(N, lh / N)) for lh in LH]
    pts = [(lh, r) for lh, r in pts if r is not None]
    print(f'{N:5}^2 ' + ' '.join(f'{dict(pts).get(lh, np.nan):6.2f}' for lh in LH))
    ax.plot(*zip(*pts), 'o-', color=c, lw=1.8, ms=5, label=f'${N}^2$, $l$ varied')
pts = [(N / 32, ratio(N, 1 / 32)) for N in NS]
ax.plot(*zip(*pts), 'ks--', lw=2.2, ms=6, mfc='none', label=r'$l = 1/32$, grid refined')
ax.axhline(1, color='0.3', lw=1.2)
ax.axvspan(0.2, 2, color='0.92', zorder=0)
ax.text(0.23, 0.33, 'Green cheaper', fontsize=10)
ax.text(30, 1.07, 'dual cheaper', fontsize=10, ha='right')
ax.set_xscale('log', base=2); ax.set_yscale('log')
ax.set_xticks(LH, [f'{v:g}' for v in LH]); ax.set_yticks([0.5, 1, 2, 4], ['0.5', '1', '2', '4']); ax.minorticks_off()
ax.set(xlim=(0.2, 40), ylim=(0.3, 5), xlabel='$l/h$ (AT1 $l$)', ylabel=r'total CG: Green / (dual $\times 2$)',
       title='AT1, B2 nine pores: Green vs dual over the whole run')
ax.grid(alpha=0.3)
ax.legend(frameon=False, fontsize=9, loc='upper left')
out = os.path.join(H, 'figures/AT1_ratio_lh.png'); fig.savefig(out, dpi=150); print(out)

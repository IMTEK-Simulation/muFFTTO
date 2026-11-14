"""TO, contrast 1e3, l = 4h: CG per solve vs grid 32^2 - 2048^2 (2048^2: dual only, NEMO 2026-11-07; 4096^2 running),
Green / Green-Jacobi / dual, slide style (raw CG, linear y from 0).  -> figures/slide_TO_2048.{png,pdf} (+ talk)"""
import csv, os, shutil
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.realpath(__file__))
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
W = {}
for r in csv.DictReader(open(os.path.join(HERE, 'figures', 'nemo', 'contrast_summary.csv'))):
    if int(float(r['contrast'])) == 1000 and int(float(r['eta_h'])) == 4:
        W[(r['precond'], int(r['N']))] = float(r['cg'])
W[('dual', 2048)] = 71.36
G = [32, 64, 128, 256, 512, 1024, 2048]
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
fig, ax = plt.subplots(figsize=(4.4, 3.3), layout='constrained')
for P, lab, col, mk in (('Green', 'Green', 'tab:green', 'o'), ('Green_Jacobi', 'Green-Jacobi', 'tab:blue', 's'),
                        ('dual', 'dual', 'tab:red', '^')):
    pts = [(n, W[(P, n)]) for n in G if (P, n) in W]
    ax.plot(*zip(*pts), '-' + mk, color=col, lw=2.2, ms=6, clip_on=False, zorder=5, label=lab)
    print(P, pts)
ax.set_xscale('log', base=2)
ax.set_xticks(G), ax.set_xticklabels([f'${n}^2$' for n in G], fontsize=10), ax.minorticks_off()
ax.set_xlim(32, 2048), ax.set_ylim(0, 350)
ax.set(xlabel='grid size', ylabel='CG iterations per solve')
ax.grid(alpha=0.3), ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
out = os.path.join(HERE, 'figures', 'slide_TO_2048')
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=150); shutil.copy(f'{out}.{ext}', TALK)

"""CG iteration study: iteration counts per case and preconditioner.

    python cg_study.py out.png case_folder_list.txt

case_folder_list.txt: lines "<case> <run folder>", case = <grid>_<physics>_<preconditioner>.
Per load step (frame): trust-region (Newton) iterations nb_outer, CG iterations (Hessian-vector
products) nb_hessp, staggered iterations, damage CG iterations cg_damage. The figure shows per load
step: CG per trust-region iteration, trust-region iterations per staggered iteration, CG of the
mechanics per load step, damage CG per staggered iteration; the table (printed, and saved as .txt)
the totals up to the common load and over the whole run.
"""
import sys
sys.path.insert(0, '/home/martin/Programming/microTopOpt/muFFTTO')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from muFFTTO import io_utils

out, listing = sys.argv[1], sys.argv[2]
cases = []
for line in open(listing):
    if line.strip():
        name, folder = line.split()
        r = io_utils.read_file(folder + '/contact_fracture.nc')
        h = r.frame_variables
        H = np.asarray(r.attributes['H_macro']).reshape(2, 2)
        i, j = np.unravel_index(np.argmax(np.abs(H)), H.shape)
        cases.append((name, (i == j) + h['lam'] * H[i, j], h))

physics_order = ['elastic', 'contact', 'damage', 'damage_contact']
labels = {'elastic': 'elasticity', 'contact': '+ contact (no damage)', 'damage': '+ damage (no contact)',
          'damage_contact': '+ damage + contact'}
colors = {'elastic': 'C0', 'contact': 'C2', 'damage': 'C1', 'damage_contact': 'C3'}

fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True)
for row, prec in enumerate(['Green', 'Green_Jacobi']):
    for name, F, h in cases:
        if not name.endswith('_' + prec) or (prec == 'Green' and name.endswith('Green_Jacobi')):
            continue
        grid = name.split('_')[0]
        phys = name[len(grid) + 1:-len(prec) - 1]
        style = '-' if grid == 'adapted' else '--'
        kw = dict(color=colors[phys], ls=style, marker='.', ms=3, lw=1,
                  label=f'{grid}, {labels[phys]}')
        stag = np.maximum(h['staggered_iterations'], 1)
        axes[row, 0].plot(F, h['nb_hessp'] / np.maximum(h['nb_outer'], 1), **kw)
        axes[row, 1].plot(F, h['nb_outer'] / stag, **kw)
        axes[row, 2].semilogy(F, np.maximum(h['nb_hessp'], 1), **kw)
        if 'damage' in phys:
            axes[row, 3].plot(F, h['cg_damage'] / stag, **kw)
    titles = ['CG iterations per trust-region iteration', 'trust-region iterations per staggered iteration',
              'CG iterations (mechanics) per load step', 'damage CG iterations per staggered iteration']
    for k, ax in enumerate(axes[row]):
        ax.set_title(f'{prec}: {titles[k]}', fontsize=10)
        ax.grid(alpha=0.3)
    axes[row, 0].legend(fontsize=7)
for ax in axes[1]:
    ax.set_xlabel(r'$\bar F$ (driven component)')
fig.tight_layout()
fig.savefig(out, dpi=100)

# table
F_common = max(F[-1] for _, F, _ in cases)          # compression: the common range ends at the largest final F
lines = [f'common load range: F from 1 down to {F_common:.4f}',
         f'{"case":42s} {"final F":>8s} {"steps":>6s} | up to common F: {"TR it":>7s} {"CG it":>8s} {"CG/TR":>6s} '
         f'{"dmg CG":>7s} | whole run: {"TR it":>7s} {"CG it":>9s} {"CG/TR":>6s} {"dmg CG":>8s} {"stag":>6s}']
for name, F, h in sorted(cases, key=lambda c: (c[0].endswith('Jacobi'), c[0])):
    m = F >= F_common - 1e-12
    tr, cg, dcg = h['nb_outer'], h['nb_hessp'], h['cg_damage']
    lines.append(f'{name:42s} {F[-1]:8.4f} {len(F):6d} | {"":16s}{int(tr[m].sum()):7d} {int(cg[m].sum()):8d} '
                 f'{cg[m].sum() / max(tr[m].sum(), 1):6.1f} {int(dcg[m].sum()):7d} | {"":11s}{int(tr.sum()):7d} '
                 f'{int(cg.sum()):9d} {cg.sum() / max(tr.sum(), 1):6.1f} {int(dcg.sum()):8d} '
                 f'{int(h["staggered_iterations"].sum()):6d}')
text = '\n'.join(lines)
print(text)
open(out.rsplit('.', 1)[0] + '.txt', 'w').write(text + '\n')
print('wrote', out)

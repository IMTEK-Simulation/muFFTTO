"""
Stage 1 plot: linear polycrystal + edge crack (Nkoumbou Kaptchouang & Gelebart 2022, Sec. 5.3).
    top:     stiffness factor per pixel and von Mises stress at the largest grid (k = 1e-6, Green), exact pixels
    bottom:  CG iterations vs grid, one panel per crack/void stiffness k; elastic work (dual counts 2) vs k
usage:  python plot_stage1.py            writes figures/polycrystal_stage1.png and prints the table
"""
import glob
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
PRECS = [('Green', 'tab:green', 'o', 1, 'Green'), ('Green_Jacobi', 'tab:blue', 's', 1, 'Green-Jacobi'),
         ('dual', 'tab:red', '^', 2, 'dual')]
runs = {}
for f in glob.glob(os.path.join(HERE, 'exp_data', 'poly_N*.npz')):
    r = np.load(f)
    runs[(int(r['N']), str(r['precond']), float(r['void_stiffness']))] = r
Ns = sorted({k[0] for k in runs})
ks = sorted({k[2] for k in runs}, reverse=True)

plt.rcParams.update({'font.size': 12})
fig = plt.figure(figsize=(18, 11), layout='constrained')
gs = fig.add_gridspec(2, len(ks) + 1)

n_max = max(Ns)
ref = runs.get((n_max, 'Green', min(ks)))
if ref is not None:
    ax = fig.add_subplot(gs[0, :2])
    im = ax.imshow(np.log10(ref['stiffness']).T, origin='lower', extent=(0, 1, 0, 1), interpolation='nearest',
                   cmap='RdYlBu_r', vmin=0, vmax=np.log10(3))
    fig.colorbar(im, ax=ax, label='log10 stiffness factor E / 1 GPa\n(crack, void layer: k, off scale)')
    ax.set_title(f'geometry, N = {n_max} (225 grains, E in [1, 3] GPa, nu = 0.3); load: E_xx')
    ax2 = fig.add_subplot(gs[0, 2:])
    vm = ref['von_mises'] / float(ref['load'])
    im2 = ax2.imshow(vm.T, origin='lower', extent=(0, 1, 0, 1), interpolation='nearest', cmap='RdYlBu_r',
                     vmin=0, vmax=np.percentile(vm, 99.5))
    fig.colorbar(im2, ax=ax2, label='von Mises stress / E_xx [GPa]')
    ax2.set_title(f'von Mises, N = {n_max}, k = {min(ks):g} (cf. paper Fig. 19)')
    for a in (ax, ax2):
        a.set_xlabel('x')
        a.set_ylabel('y')

print(f'{"N":>5s} {"k":>7s} | ' + ' '.join(f'{lab:>13s}' for _, _, _, _, lab in PRECS) + ' | sigma_xx/E_xx')
for col, k in enumerate(ks):
    ax = fig.add_subplot(gs[1, col])
    for p, c, m, w, lab in PRECS:
        its = [int(runs[(n, p, k)]['cg_its']) if (n, p, k) in runs else np.nan for n in Ns]
        ax.plot(Ns, its, marker=m, color=c, lw=1.8, label=lab)
    ax.set_xticks(Ns, [f'{n}²' for n in Ns])
    ax.set_ylim(0, None)
    ax.set_xlabel('grid')
    ax.set_ylabel('CG iterations (energy, 1e-6)')
    ax.set_title(f'crack / void stiffness k = {k:g}')
    ax.grid(True, color='0.9')
    if col == 0:
        ax.legend(frameon=False)
    for n in Ns:
        row = [runs[(n, p, k)] for p, *_ in PRECS if (n, p, k) in runs]
        print(f'{n:5d} {k:7.0e} | ' + ' '.join(f'{int(r["cg_its"]):13d}' for r in row) +
              f' | {float(row[0]["sigma"][0, 0]) / float(row[0]["load"]):.5f}')
ax = fig.add_subplot(gs[1, len(ks)])
for p, c, m, w, lab in PRECS:
    work = [w * int(runs[(n_max, p, k)]['cg_its']) if (n_max, p, k) in runs else np.nan for k in ks]
    ax.plot(range(len(ks)), work, marker=m, color=c, lw=1.8, label=lab)
ax.set_xticks(range(len(ks)), [f'{k:g}' for k in ks])
ax.set_ylim(0, None)
ax.set_xlabel('crack / void stiffness k')
ax.set_ylabel('elastic work (K applications, dual 2 per it.)')
ax.set_title(f'cost at N = {n_max}')
ax.grid(True, color='0.9')
fig.suptitle('Stage 1: linear polycrystal with an edge crack (Nkoumbou Kaptchouang & Gelebart 2022, Sec. 5.3) '
             '- elastic preconditioners', fontsize=14)
out = os.path.join(HERE, 'figures', 'polycrystal_stage1.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print(f'wrote {out}')

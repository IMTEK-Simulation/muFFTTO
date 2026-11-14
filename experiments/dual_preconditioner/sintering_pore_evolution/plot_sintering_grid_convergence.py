"""Grid convergence of the sintering model's physical output (not the solver cost): quantities at the end of the run
(t = 495 us) and over time, against the grid 32^2 - 1024^2, W = 10 nm (solid) and W = 4 nm (dashed). Green-Jacobi
runs (the only preconditioner run on every grid; the physics does not depend on the preconditioner).
  top:    strain E_yy under the 50 MPa uniaxial load, normalised stiffness C_yy / C_solid
  bottom: free-surface length, grain-boundary length (2D: lengths in nm)
    python plot_sintering_grid_convergence.py  ->  figures/sintering_grid_convergence.png
"""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDS = [32, 64, 128, 256, 512, 1024]
CASES = [('W10_P50u_k0.0001_W10', '-', r'$W = 10$ nm'), ('W4_P50u_k0.0001', '--', r'$W = 4$ nm')]
cmap = plt.get_cmap('viridis')
col = {N: cmap(i / (len(GRIDS) - 1)) for i, N in enumerate(GRIDS)}
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
fig, axs = plt.subplots(2, 3, figsize=(15, 8.2), layout='constrained')
QUANT = [('E', r'strain $E_{yy}$ under $-50$ MPa [$10^{-3}$]', lambda z: z['E'][:, 1] * 1e3),
         ('C', r'stiffness $C_{yy} / C_{\rm solid}$', lambda z: z['C_eff'][:, 1, 1]),
         ('s', 'free-surface length [nm]', lambda z: z['surface_length']),
         ('g', 'grain-boundary length [nm]', lambda z: z['gb_length'])]
end = {q[0]: {c[0]: [] for c in CASES} for q in QUANT}
for pat, ls, lab in CASES:
    for N in GRIDS:
        f = os.path.join(HERE, 'exp_data', f'SINT_N{N}_Green_Jacobi_{pat}_steps.npz')
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        if z['t'][-1] < 4.9e-4:
            continue
        for key, _, fun in QUANT:
            end[key][pat].append((N, float(fun(z)[-1])))
        if pat.startswith('W10'):                            # time histories, W = 10 nm only
            t = z['t'] * 1e6
            axs[0, 0].plot(t, QUANT[0][2](z), color=col[N], lw=1.8, label=f'${N}^2$')
            axs[1, 0].plot(t, QUANT[2][2](z), color=col[N], lw=1.8)
for ax, key, ylab in ((axs[0, 1], 'E', QUANT[0][1]), (axs[0, 2], 'C', QUANT[1][1]),
                      (axs[1, 1], 's', QUANT[2][1]), (axs[1, 2], 'g', QUANT[3][1])):
    for pat, ls, lab in CASES:
        if end[key][pat]:
            ax.plot(*zip(*end[key][pat]), ls + 'o', color='k', mfc='k' if ls == '-' else 'white', lw=2, ms=6,
                    label=lab)
    ax.set_xscale('log', base=2)
    ax.set_xticks(GRIDS), ax.set_xticklabels([f'${n}^2$' for n in GRIDS]), ax.minorticks_off()
    ax.set(xlabel='grid size', ylabel=ylab + r' at $t = 495\,\mu$s')
    ax.grid(alpha=0.3)
axs[0, 1].legend()
axs[0, 0].set(xlabel=r'time [$\mu$s]', ylabel=QUANT[0][1]), axs[0, 0].legend(title=r'$W = 10$ nm', fontsize=10, ncol=2)
axs[1, 0].set(xlabel=r'time [$\mu$s]', ylabel=QUANT[2][1])
for ax in axs[:, 0]:
    ax.grid(alpha=0.3)
fig.suptitle(r'Sintering model: grid convergence of the physical response (Green-Jacobi runs, $-50$ MPa uniaxial, '
             r'$k_{\rm res} = 10^{-4}$)', fontsize=15)
out = os.path.join(HERE, 'figures', 'sintering_grid_convergence.png')
fig.savefig(out, dpi=130)
for key, _, _ in QUANT:
    for pat, _, lab in CASES:
        print(key, pat[:3], [(n, round(v, 4)) for n, v in end[key][pat]])
print('wrote', out)

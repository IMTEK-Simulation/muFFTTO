"""How the closed void pockets make the 64^3 sine-start design (a = 5, w = 0.2) auxetic: uniaxial compression along y
with free lateral faces (macro stress sigma = -e_y e_y), for the converged design with its 2 pockets and for the same
design with the pockets filled (data/pocket_fill_test, make_TO3D_deformation.py: the 6 unit load cases).
Macro strain E = C_eff^-1 sigma (Voigt, engineering shear = the load-case weights), displacement
x = X + s (E X + sum_k E_k u~_k), magnified by s (same for both designs).
Row per design: the x-y cut through the pocket (z = index with the largest pocket area), undeformed and deformed, rho on
its exact pixels mapped onto the deformed grid; dashed: the undeformed cell, solid: the deformed cell outline.
    python plot_pocket_auxetic_mechanism.py  ->  figures/pocket_auxetic_mechanism.png
"""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from scipy import ndimage

from elastic_properties import properties, voigt_from_stresses

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(HERE, 'data', 'pocket_fill_test')
VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def load(name):
    with netCDF4.Dataset(os.path.join(T, f'sine64_{name}_deformation.nc')) as d:
        rho = np.array(d['phase_field'][0, 0])
        U = np.array(d['displacement'][:, :, 0])                  # (6 load cases, 3, N, N, N), nodal
        hs = np.array(d['homogenized_stress'])
    C = voigt_from_stresses(hs)
    sig = np.zeros(6)
    sig[1] = -1.0                                                 # uniaxial compression along y, unit stress
    Ev = np.linalg.solve(C, sig)
    E = np.zeros((3, 3))
    for k, (i, j) in enumerate(VOIGT):
        E[i, j] = E[j, i] = Ev[k] if i == j else 0.5 * Ev[k]
    u = np.tensordot(Ev, U, axes=(0, 0))                          # fluctuation (3, N, N, N), in cell units?
    return rho, E, u, properties(C)


def pocket_mask(rho):
    m = rho < 0.5
    lab, n = ndimage.label(m)
    N = rho.shape[0]
    par = list(range(n + 1))

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for ax in range(3):
        a, b = np.take(lab, 0, axis=ax), np.take(lab, N - 1, axis=ax)
        mm = (a > 0) & (b > 0)
        for x, y in zip(a[mm], b[mm]):
            par[f(x)] = f(y)
    root = np.array([f(k) for k in range(n + 1)])[lab]
    root[~m] = 0
    ids, cnt = np.unique(root[root > 0], return_counts=True)
    return (root > 0) & (root != ids[np.argmax(cnt)])


rho0, E0, u0, p0 = load('original')
rhoF, EF, uF, pF = load('filled')
N = rho0.shape[0]
pk = pocket_mask(rho0)
SCALE = 0.2 / abs(E0[1, 1])            # magnification (same for both): the design with pockets shown at -20 % axial strain
kz = int(np.argmax(pk.sum(axis=(0, 1))))                          # x-y cut with the largest pocket area
print('cut z index', kz, 'pocket pixels in cut', int(pk[:, :, kz].sum()))
print('macro strain original E_xx, E_yy, E_zz =', np.round(np.diag(E0), 4), ' filled =', np.round(np.diag(EF), 4))
# max |u~| unit check: displacement stored in grid points (muEye convention) -> cell units
grid_units = np.abs(u0).max() > 1.5
to_cell = (1.0 / N) if grid_units else 1.0

plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
fig, axs = plt.subplots(2, 2, figsize=(11.5, 11), layout='constrained')
X, Y = np.meshgrid(np.arange(N + 1) / N, np.arange(N + 1) / N, indexing='ij')   # pixel corners
for row, (lab, rho, E, u, p) in enumerate((('with the 2 closed pockets (converged design)', rho0, E0, u0, p0),
                                            ('pockets filled with solid', rhoF, EF, uF, pF))):
    sl = rho[:, :, kz]
    # node displacement on the cut (periodic: corner N = corner 0)
    ux = np.pad(u[0, :, :, kz], ((0, 1), (0, 1)), mode='wrap') * to_cell
    uy = np.pad(u[1, :, :, kz], ((0, 1), (0, 1)), mode='wrap') * to_cell
    xd = X + SCALE * (E[0, 0] * X + E[0, 1] * Y + ux)
    yd = Y + SCALE * (E[1, 0] * X + E[1, 1] * Y + uy)
    for col, (xx, yy, title) in enumerate(((X, Y, 'undeformed'), (xd, yd, r'compressed along $y$ (magnified)'))):
        ax = axs[row, col]
        ax.pcolormesh(xx, yy, sl, cmap='Greys', vmin=0, vmax=1, shading='flat', rasterized=True)
        if row == 0:
            pc = np.ma.masked_where(~pk[:, :, kz], np.ones_like(sl))
            ax.pcolormesh(xx, yy, pc, cmap='autumn', vmin=0, vmax=2, shading='flat', alpha=0.85, rasterized=True)
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k--', lw=1.2)
        if col == 1:
            cx = [xd[0, 0], xd[-1, 0], xd[-1, -1], xd[0, -1], xd[0, 0]]
            cy = [yd[0, 0], yd[-1, 0], yd[-1, -1], yd[0, -1], yd[0, 0]]
            mx = SCALE * np.array([E[0, 0], E[1, 1]])
            ax.annotate('', xy=(1 + mx[0], 0.5), xytext=(1.0, 0.5),
                        arrowprops=dict(arrowstyle='->', color='tab:blue' if mx[0] < 0 else 'tab:red', lw=2.5))
            ax.annotate('', xy=(0 - mx[0], 0.5), xytext=(0.0, 0.5),
                        arrowprops=dict(arrowstyle='->', color='tab:blue' if mx[0] < 0 else 'tab:red', lw=2.5))
            for yy_ in (1.08, -0.08):
                ax.annotate('', xy=(0.5, yy_ + (-0.07 if yy_ > 1 else 0.07)), xytext=(0.5, yy_),
                            arrowprops=dict(arrowstyle='->', color='k', lw=2.5))
            ax.text(0.5, -0.2, rf'lateral strain $E_{{xx}} = {E[0, 0] / abs(E[1, 1]):+.2f}\,|E_{{yy}}|$: '
                    + ('contracts sideways (auxetic)' if E[0, 0] < 0 else 'bulges sideways (not auxetic)'),
                    ha='center', fontsize=13, transform=ax.transData)
        ax.set_aspect('equal'), ax.set_xlim(-0.25, 1.25), ax.set_ylim(-0.25, 1.25)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f'{lab}: {title}' if col == 0 else title, fontsize=13)
    axs[row, 0].text(-0.22, 0.5, rf'$\nu_{{21}} = {p["nu21"]:+.2f}$', rotation=90, va='center', fontsize=16,
                     transform=axs[row, 0].transData)
fig.suptitle(rf'64$^3$ sine-start design, cut $z = {kz}/64$ through a pocket (orange): uniaxial compression along $y$, '
             r'free sides', fontsize=15)
out = os.path.join(HERE, 'figures', 'pocket_auxetic_mechanism.png')
fig.savefig(out, dpi=120)
print('wrote', out, 'grid-unit displacements:', grid_units)

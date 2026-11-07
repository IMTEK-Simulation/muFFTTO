"""
Reference data of the periodic AT2 benchmark (see benchmark.tex) from the fixed-l runs of
example_2D_phase_field_fracture_AT2.py (l = 1/32, grids 32^2 - 1024^2, dual preconditioner; all three
preconditioners give the same curves to the printed digits).

    B0  homogeneous cell (no inclusions): analytic, code verification
    B1  one centred circular pore, r = 0.2 (soft phase 1e-3, damage solved in it)
    B1h the same as a true hole (zero stiffness, no damage field; Green-Jacobi)
    B2  nine polydisperse pores (disks.csv)
    B3  the same nine disks as stiff (10x), tough inclusions

Writes into data/:  disks.csv, B0_homogeneous.csv, B<k>_N<n>_curve.csv (E_xx, sigma_xx), reference.csv
(peak stress, strain at peak, crack energy of the final state per case and grid) and figures/benchmark.png.

usage:  python make_benchmark.py
"""
import csv
import glob
import os
import sys

import numpy as np
import netCDF4
import matplotlib
import matplotlib.ticker

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '../../../..')))
from muFFTTO import geometry  # noqa: E402

EXP = os.path.join(HERE, '..', 'exp_data')
DATA = os.path.join(HERE, 'data')
FIG = os.path.join(HERE, 'figures')
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# model parameters of the runs (lambda = K_3D - G with E = 1, nu = 0.2: plane strain with E' = 0.9375, nu' = 0.125)
lam, mu = 5.0 / 36.0, 5.0 / 12.0
Gc, ell, k_res = 5e-4, 1.0 / 32.0, 1e-4
CASES = {'B1': 'AT2_N{n}_dual_l0.03125.nc',
         'B1h': 'AT2_N{n}_Green_Jacobi_holes_l0.03125.nc',
         'B2': 'AT2_N{n}_dual_polydisperse_pores_l0.03125.nc',
         'B3': 'AT2_N{n}_dual_polydisperse_pores_incl10_l0.03125{suffix}.nc'}
TITLES = {'B1': 'B1  one pore', 'B1h': 'B1h  one hole', 'B2': 'B2  nine pores', 'B3': 'B3  nine stiff inclusions (10x)'}

# ---------------------------------------------------------------------------------------------------------------------
# disks
centres, radii = geometry.polydisperse_disk_centres_and_radii(2, 0.25, (0.05, 0.14), 0.04, 0)
with open(os.path.join(DATA, 'disks.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['x', 'y', 'r'])
    for c, r in zip(centres, radii):
        w.writerow([f'{c[0]:.10f}', f'{c[1]:.10f}', f'{r:.10f}'])

# ---------------------------------------------------------------------------------------------------------------------
# B0: homogeneous cell under E = E_xx e_x x e_x. All of psi_0 = M/2 E^2 (M = lambda + 2 mu) is tensile in the Amor split
# (K_2D = lambda + mu), the damage is uniform: d = M E^2 / (Gc/l + M E^2) (history = monotone load), and
# sigma_xx = ((1 - d)^2 + k) M E. Without k: E_c = sqrt(Gc / (3 l M)), sigma_c = 9/16 sqrt(M Gc / (3 l)).
M = lam + 2.0 * mu
E0 = np.linspace(0.0, 0.3, 30001)
d0 = M * E0 ** 2 / (Gc / ell + M * E0 ** 2)
s0 = ((1.0 - d0) ** 2 + k_res) * M * E0
np.savetxt(os.path.join(DATA, 'B0_homogeneous.csv'), np.c_[E0[::100], s0[::100], d0[::100]], delimiter=',',
           header='E_xx,sigma_xx,d', comments='', fmt='%.10e')
i0 = s0.argmax()
print(f'B0: sigma_c = {s0[i0]:.6e} at E_xx = {E0[i0]:.5f}  (k = 0: {9 / 16 * np.sqrt(M * Gc / (3 * ell)):.6e} at '
      f'{np.sqrt(Gc / (3 * ell * M)):.5f})')


# ---------------------------------------------------------------------------------------------------------------------
def crack_energy(d, domain=None):
    """Gc/2 int d^2/l + l |grad d|^2 of a nodal periodic Q1 field d[x, y] on [0, 1)^2, exact (2x2 Gauss);
    domain (0/1 per pixel) restricts the integral, e.g. to the matrix around holes."""
    n = d.shape[0]
    h = 1.0 / n
    d00, d10, d01, d11 = d, np.roll(d, -1, 0), np.roll(d, -1, 1), np.roll(np.roll(d, -1, 0), -1, 1)
    g = (1.0 - 1.0 / np.sqrt(3.0)) / 2.0
    total = 0.0
    for a in (g, 1.0 - g):
        for b in (g, 1.0 - g):
            val = d00 * (1 - a) * (1 - b) + d10 * a * (1 - b) + d01 * (1 - a) * b + d11 * a * b
            dx = ((d10 - d00) * (1 - b) + (d11 - d01) * b) / h
            dy = ((d01 - d00) * (1 - a) + (d11 - d10) * a) / h
            total += np.sum((1.0 if domain is None else domain) * (val ** 2 / ell + ell * (dx ** 2 + dy ** 2))) \
                * h * h / 4.0
    return 0.5 * Gc * total


rows, finest = [], {}
for case, pattern in CASES.items():
    for n in (32, 64, 128, 256, 512, 1024):
        files = [os.path.join(EXP, pattern.format(n=n, suffix=s)) for s in ('', '_lh')]
        files = [f for f in files if os.path.exists(f)]
        if not files:
            continue
        with netCDF4.Dataset(files[0]) as ds:
            load = np.asarray(ds['load'][:])
            sxx = np.asarray(ds['macro_stress'][:, 0, 0])
            d_end = np.asarray(ds['damage'][-1, 0])
        # the 1024^2 B2 run stopped before the cell broke; its restart continues the load path
        restart = files[0].replace('.nc', '_restart.nc')
        if os.path.exists(restart):
            with netCDF4.Dataset(restart) as ds:
                keep = load < np.asarray(ds['load'][0]) - 1e-12
                load = np.r_[load[keep], np.asarray(ds['load'][:])]
                sxx = np.r_[sxx[keep], np.asarray(ds['macro_stress'][:, 0, 0])]
                d_end = np.asarray(ds['damage'][-1, 0])
        broken = sxx[-1] < 0.2 * sxx.max()
        i = sxx.argmax()
        # holes: the matrix pixels only (a pixel takes the phase of its lower-left node, as in the solver)
        x = np.arange(n) / n
        r2 = ((x[:, None] - 0.5 + 0.5) % 1.0 - 0.5) ** 2 + ((x[None, :] - 0.5 + 0.5) % 1.0 - 0.5) ** 2
        G = crack_energy(d_end, (r2 >= 0.2 ** 2).astype(float) if case == 'B1h' else None)
        rows.append([case, n, ell * n, f'{sxx[i]:.6e}', f'{load[i]:.3f}', f'{G:.6e}' if broken else '',
                     f'{G / Gc:.4f}' if broken else '', f'{load[-1]:.3f}', int(broken)])
        np.savetxt(os.path.join(DATA, f'{case}_N{n}_curve.csv'), np.c_[np.r_[0, load], np.r_[0, sxx]], delimiter=',',
                   header='E_xx,sigma_xx', comments='', fmt='%.10e')
        finest[case] = (n, d_end)
        print(f'{case} N {n:5d}  l/h {ell * n:5.0f}  peak {sxx[i]:.5e} at {load[i]:.3f}  '
              f'Gamma/Gc {G / Gc:.4f}{"" if broken else " (not broken)"}  last E {load[-1]:.3f}')

with open(os.path.join(DATA, 'reference.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['case', 'N', 'l_over_h', 'peak_sigma_xx', 'E_xx_at_peak', 'crack_energy_final', 'crack_length_final',
                'E_xx_final', 'broken'])
    w.writerows(rows)

# ---------------------------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(3, 4, figsize=(20, 14), layout='constrained', dpi=150)
for col, case in enumerate(CASES):
    n, d = finest[case]
    ax = axes[0, col]
    ax.imshow(d.T, origin='lower', vmin=0, vmax=1, cmap='inferno', interpolation='nearest', extent=(0, 1, 0, 1))
    t = np.linspace(0, 2 * np.pi, 200)
    disks = [((0.5, 0.5), 0.2)] if case.startswith('B1') else list(zip(centres, radii))
    for (cx, cy), r in disks:                               # periodic images inside the cell
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                ax.plot(cx + sx + r * np.cos(t), cy + sy + r * np.sin(t), '-', lw=0.6, color='c')
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f'{TITLES[case]}\nfinal damage, {n}$^2$ (l/h = {ell * n:g})', fontsize=10)
    case_rows = [r for r in rows if r[0] == case]
    for r in case_rows:
        c = np.loadtxt(os.path.join(DATA, f'{case}_N{r[1]}_curve.csv'), delimiter=',', skiprows=1)
        axes[1, col].plot(c[:, 0], c[:, 1], '-', lw=1.0, label=f'{r[1]}$^2$ (l/h = {r[2]:g})')
    axes[1, col].plot(E0, s0, ':', color='0.5', lw=1, label='B0 homogeneous')
    axes[1, col].set_xlim(0, 0.08), axes[1, col].set_ylim(0, None)
    axes[1, col].set_xlabel(r'$\bar\varepsilon_{xx}$'), axes[1, col].set_ylabel(r'$\bar\sigma_{xx}$')
    axes[1, col].grid(True, color='0.9'), axes[1, col].legend(fontsize=8)
    h = np.array([1.0 / r[1] for r in case_rows])
    axes[2, col].plot(h, [float(r[3]) for r in case_rows], 'o-', color='C3', label='peak $\\bar\\sigma_{xx}$')
    axes[2, col].set_xscale('log'), axes[2, col].set_xlabel('h'), axes[2, col].set_ylabel('peak stress', color='C3')
    axes[2, col].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[2, col].grid(True, color='0.9', which='both')
    tw = axes[2, col].twinx()
    hb = [(1.0 / r[1], float(r[6])) for r in case_rows if r[6]]
    if hb:
        tw.plot(*zip(*hb), 's--', color='C0')
    tw.set_ylabel(r'crack length $\Gamma / G_c$ (final)', color='C0')
out = os.path.join(FIG, 'benchmark.png')
fig.savefig(out)
print(f'wrote {out}')

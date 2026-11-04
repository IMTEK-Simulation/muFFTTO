"""
Damage profile across the crack against the 1D AT2 profile d = exp(-|x| / l). Benchmark B1 (one centred pore, r = 0.2,
fixed l = 1/32, final state E_xx = 0.08, dual runs AT2_N<n>_dual_l0.03125): the crack is vertical at x ~ 0.5; the
profile is the row y = 0 of nodes (farthest from the pore), shifted to the node of maximal damage.

AT2 has no elastic threshold: the whole cell carries a background damage d_bg (history of the peak load), so the
profile is compared both with exp(-|xi|) and with d_bg + (1 - d_bg) exp(-|xi|), xi = (x - x_0) / l. The decay length
is fitted from ln((d - d_bg) / (1 - d_bg)) on 0.5 <= |xi| <= 3.

usage:  python plot_AT2_damage_profile.py      writes figures/AT2_damage_profile.png and prints the fitted decay lengths
"""
import os

import numpy as np
import netCDF4
import matplotlib
import matplotlib.ticker

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
ELL = 1.0 / 32.0
GRIDS = (32, 64, 128, 1024)
COLORS = {32: 'C7', 64: 'C1', 128: 'C2', 1024: 'C0'}

fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout='constrained', dpi=150)
xi_ref = np.linspace(-6, 6, 1201)
for n in GRIDS:
    with netCDF4.Dataset(os.path.join(HERE, 'exp_data', f'AT2_N{n}_dual_l0.03125.nc')) as ds:
        row = np.asarray(ds['damage'][-1, 0])[:, 0]                  # d[x, y = 0]
    i0 = int(row.argmax())
    row = np.roll(row, n // 2 - i0)                                  # crack node to the middle
    xi = (np.arange(n) - n // 2) / n / ELL
    d_bg = row[0]                                                    # |xi| = 16: far from the crack
    lh = ELL * n
    style = dict(color=COLORS[n], lw=1.5, ms=5 if n < 1024 else 0, marker='o')
    axes[0].plot(xi, row, '-', label=f'{n}$^2$ (l/h = {lh:g})', **style)
    side = xi >= 0
    y = (row[side] - d_bg) / (1.0 - d_bg)
    axes[1].plot(xi[side], np.where(y > 0, y, np.nan), '-', **style)
    fit = side & (xi >= 0.5) & (xi <= 3.0)
    yf = (row[fit] - d_bg) / (1.0 - d_bg)
    slope = np.polyfit(xi[fit], np.log(yf), 1)[0]
    print(f'N {n:5d}  l/h {lh:5g}  d_max {row.max():.3f}  d_bg {d_bg:.3f}  fitted decay length {-1 / slope:.3f} l')
d_bg = 0.095
axes[0].plot(xi_ref, np.exp(-np.abs(xi_ref)), 'k--', lw=1.5, label=r'$\exp(-|\xi|)$')
axes[0].plot(xi_ref, d_bg + (1 - d_bg) * np.exp(-np.abs(xi_ref)), 'k:', lw=1.5,
             label=r'$d_{bg} + (1 - d_{bg}) \exp(-|\xi|)$, $d_{bg}$ = 0.095')
axes[1].plot(xi_ref[xi_ref >= 0], np.exp(-xi_ref[xi_ref >= 0]), 'k--', lw=1.5)
axes[0].set_xlim(-6, 6), axes[0].set_ylim(0, 1.05)
axes[0].set_xlabel(r'$\xi = (x - x_0) / l$', fontsize=12), axes[0].set_ylabel('damage d', fontsize=12)
axes[0].set_title('B1, final damage across the crack (row y = 0, l = 1/32)', fontsize=13)
axes[1].set_yscale('log'), axes[1].set_xlim(0, 6), axes[1].set_ylim(1e-3, 1.2)
axes[1].yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.001, 0.01, 0.1, 1]))
axes[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
axes[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[1].set_xlabel(r'$\xi = (x - x_0) / l$', fontsize=12)
axes[1].set_ylabel(r'$(d - d_{bg}) / (1 - d_{bg})$', fontsize=12)
axes[1].set_title(r'decay: a straight line = exponential; dashed = $\exp(-\xi)$', fontsize=13)
for ax in axes:
    ax.tick_params(labelsize=12), ax.grid(True, color='0.9')
fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=6, fontsize=11)
out = os.path.join(HERE, 'figures', 'AT2_damage_profile.png')
fig.savefig(out)
print(f'wrote {out}')

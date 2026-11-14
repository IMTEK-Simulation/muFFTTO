"""
Stage 2 analysis: 2-grain chain in a contracting cell (rho[x, y], y = s Y).
  L = s L0 (cell height = chain period), neck = width of rho > 0.5 on the GB row, s <rho> = mass (conserved),
  sigma_gb = -<sigma_yy>_GB from the elastic solve, compared with force balance P L_x / neck.
Writes figures/chain_pressure.png, _snapshots.png, _rate.png
"""
import glob
import os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
base = 'SINT_N96_Green_W4_P{}u_k0.0001_chain_{}'
cases = [('no RB, 0 MPa', 0, 'noRB'), ('RB, 0 MPa', 0, 'P0'), ('RB, 25 MPa', 25, 'P25'), ('RB, 50 MPa', 50, 'P50'),
         ('RB, 100 MPa', 100, 'P100')]
L0 = 90.24
colors = ['#888888', '#111111', '#1f77b4', '#d95f02', '#7b3294']
styles = [':', '-', '--', '-.', '-']


def crossings(line):
    s = line - 0.5
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    return idx + s[idx] / (s[idx] - s[idx + 1])


def measure(run):
    files = sorted(glob.glob(os.path.join(here, 'exp_data', run + '_snaps', 'step_*.npz')), key=lambda f: int(os.path.basename(f)[5:-4]))
    h = L0 / np.load(files[0])['rho'].shape[0]
    out = {k: [] for k in ('t', 's', 'neck', 'mass', 'sigma', 'V', 'rho_max', 'eta_min_area')}
    for f in files:
        d = np.load(f)
        r = d['rho'].astype(float)
        neck = crossings(r[:, r.shape[1] // 2])
        out['t'].append(float(d['t']))
        out['s'].append(float(d['s']))
        out['neck'].append((neck.max() - neck.min()) * h)
        out['mass'].append(float(d['s']) * r.mean())
        out['sigma'].append(float(d['sigma_gb']))
        out['V'].append(float(d['V']))
        out['rho_max'].append(r.max())
        out['eta_min_area'].append(min(np.sum(r * e) for e in d['etas'].astype(float)))
    return {k: np.array(v) for k, v in out.items()}, files


res = []
for label, p, tag in cases:
    m, files = measure(base.format(p, tag))
    res.append((label, p, m, files))
    a = m['eta_min_area']
    print(f'{label:13s} L {L0 * m["s"][0]:.2f} -> {L0 * m["s"][-1]:.2f} nm, neck {m["neck"][0]:.1f} -> {m["neck"][-1]:.1f}, '
          f'mass drift {m["mass"][-1] / m["mass"][0] - 1:.2e}, rho_max {m["rho_max"].max():.3f}, '
          f'smallest grain {a.min():.0f}/{a[0]:.0f}')

plt.rcParams.update({'font.size': 13})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
for (label, p, m, _), c, ls in zip(res, colors, styles):
    t = m['t'] * 1e6
    ax[0].plot(t, L0 * m['s'], color=c, ls=ls, lw=2, label=label)
    ax[1].plot(t, m['neck'], color=c, ls=ls, lw=2)
    if p > 0:
        w = m['t'] > 2e-5
        ax[2].plot(t[w], m['sigma'][w] / 1e6, color=c, ls=ls, lw=2)
        ax[2].plot(t[w], p * L0 / m['neck'][w], color=c, ls=':', lw=1.5)
for a, yl in zip(ax, ('chain period L [nm]', 'neck width [nm]', 'GB normal stress [MPa]\n(solid/dashed: elastic solve, dotted: force balance)')):
    a.set_xlabel(r'time [$\mu$s]')
    a.set_ylabel(yl)
    a.set_xlim(left=0)
    a.grid(alpha=0.3)
ax[2].set_ylim(bottom=0)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc='outside upper center', ncol=len(res))
fig.savefig(os.path.join(here, 'figures', 'chain_pressure.png'), dpi=150)

# equal-neck rates
all_rb = [r for r in res if r[0].startswith('RB')]
targets = [30.0, 34.0, 42.0]                               # 42 nm is not reached at P = 0 (nan)
print('shrinkage rate -dL/dt [nm/ms] at equal neck width; P =', [p for _, p, _, _ in all_rb])
table = {}
for n in targets:
    row = []
    for _, p, m, _ in all_rb:
        w = (np.abs(m['neck'] - n) <= 2.0) & (m['t'] > 5e-5)
        row.append(-np.polyfit(m['t'][w] * 1e3, L0 * m['s'][w], 1)[0] if w.sum() >= 2 else np.nan)
    table[n] = row
    print(f'neck {n:4.0f}: ' + '  '.join(f'{v:7.2f}' for v in row))
fig, a = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
for n, c, mk in zip(targets, ('#444444', '#1f77b4', '#d95f02', '#7b3294'), 'os^D'):
    a.plot([p for _, p, _, _ in all_rb], table[n], color=c, marker=mk, ms=8, lw=2, label=f'neck width {n:g} nm')
a.set_xlabel('applied pressure [MPa]')
a.set_ylabel('chain shrinkage rate [nm/ms]')
a.set_xlim(left=0)
a.set_ylim(bottom=0)
a.grid(alpha=0.3)
a.legend()
fig.savefig(os.path.join(here, 'figures', 'chain_pressure_rate.png'), dpi=150)

# density snapshots in physical coordinates: exact (rectangular) pixels
fig, axes = plt.subplots(3, 3, figsize=(11, 11), constrained_layout=True)
for r, (label, p, m, files) in enumerate([res[1], res[3], res[4]]):
    for c, i in enumerate((0, len(files) // 2, len(files) - 1)):
        d = np.load(files[i])
        Ly = L0 * float(d['s'])
        im = axes[r, c].imshow(d['rho'].T, origin='lower', cmap='gray_r', vmin=0, vmax=1, interpolation='nearest',
                               extent=(0, L0, 0, Ly))
        axes[r, c].set_ylim(0, L0)
        axes[r, c].set_title(f'{label}, t = {float(d["t"]) * 1e6:g} $\\mu$s')
        axes[r, c].set_xlabel('x [nm]')
        axes[r, c].set_ylabel('y [nm]')
fig.colorbar(im, ax=axes, label=r'$\rho$', shrink=0.7)
fig.savefig(os.path.join(here, 'figures', 'chain_pressure_snapshots.png'), dpi=150)

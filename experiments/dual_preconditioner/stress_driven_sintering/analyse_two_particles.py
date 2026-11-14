"""
Two-particle diagnostics from the snapshots of sintering_phase_field.py (rho[x, y], etas[i, x, y]; line of centres
along y, grain boundary on the middle row).

  centre distance  y-distance of the centroids of rho * eta_i
  tip distance     distance of the outer rho = 0.5 crossings on the column through the centres (macroscopic length)
  neck width       width of rho > 0.5 on the grain-boundary row (linear interpolation of the crossing)
  mass             mean rho

    python analyse_two_particles.py <fig_name> <label=run_name> [<label=run_name> ...]
Writes exp_data/<run_name>_two_particle.npz, figures/<fig_name>.png
"""
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))


def crossings(line):
    """fractional indices of all rho = 0.5 crossings of a 1D (non-periodic view) profile"""
    s = line - 0.5
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    return idx + s[idx] / (s[idx] - s[idx + 1])


def measure(run):
    files = sorted(glob.glob(os.path.join(here, 'exp_data', run + '_snaps', 'step_*.npz')), key=lambda f: int(os.path.basename(f)[5:-4]))
    h = float(np.load(os.path.join(here, 'exp_data', run + '_steps.npz'))['h'])
    out = {k: [] for k in ('t', 'centre', 'tip', 'neck', 'mass', 'rho_min', 'rho_max')}
    for f in files:
        d = np.load(f)
        rho, etas = d['rho'].astype(float), d['etas'].astype(float)
        n = rho.shape[0]
        y = np.arange(n) * h
        cy = [np.sum(rho * e * y[None, :]) / np.sum(rho * e) for e in etas]
        tips = crossings(rho[n // 2, :])
        neck = crossings(rho[:, n // 2])
        out['t'].append(float(d['t']))
        out['centre'].append(abs(cy[1] - cy[0]))
        out['tip'].append((tips.max() - tips.min()) * h)
        out['neck'].append((neck.max() - neck.min()) * h)
        out['mass'].append(rho.mean())
        out['rho_min'].append(rho.min())
        out['rho_max'].append(rho.max())
    return {k: np.array(v) for k, v in out.items()}, files


fig_name = sys.argv[1]
runs = [a.rsplit('=', 1) for a in sys.argv[2:]]
plt.rcParams.update({'font.size': 13})
shades = ['#111111', '#1f77b4', '#d95f02', '#7b3294']
styles = ['-', '--', '-.', ':']
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
results = []
for k, (label, run) in enumerate(runs):
    m, files = measure(run)
    results.append((label, m, files))
    np.savez(os.path.join(here, 'exp_data', run + '_two_particle.npz'), **m)
    t = m['t'] * 1e6
    kw = dict(color=shades[k], ls=styles[k], lw=2, label=label)
    axes[0].plot(t, m['centre'] - m['centre'][0], **kw)
    axes[1].plot(t, m['tip'] - m['tip'][0], **kw)
    axes[2].plot(t, m['neck'], **kw)
    print(f'{label}: centre {m["centre"][0]:.3f} -> {m["centre"][-1]:.3f} nm, tip {m["tip"][0]:.3f} -> '
          f'{m["tip"][-1]:.3f} nm, neck {m["neck"][0]:.2f} -> {m["neck"][-1]:.2f} nm, '
          f'mass drift {m["mass"][-1] / m["mass"][0] - 1:.2e}, '
          f'rho [{m["rho_min"].min():.3f}, {m["rho_max"].max():.3f}]')
for ax, yl in zip(axes, ('change of centre distance [nm]', 'change of tip-to-tip length [nm]', 'neck width [nm]')):
    ax.set_xlabel(r'time [$\mu$s]')
    ax.set_ylabel(yl)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.3)
    ax.ticklabel_format(style='plain', useOffset=False)
handles, labels = axes[0].get_legend_handles_labels()
if len(runs) > 1:
    fig.legend(handles, labels, loc='outside upper center', ncol=len(runs))
fig.savefig(os.path.join(here, 'figures', fig_name + '.png'), dpi=150)

# density snapshots (exact pixels), first / middle / last, one row per run
fig, axes = plt.subplots(len(runs), 3, figsize=(11, 3.9 * len(runs)), squeeze=False, constrained_layout=True)
for r, (label, m, files) in enumerate(results):
    for c, i in enumerate((0, len(files) // 2, len(files) - 1)):
        d = np.load(files[i])
        ax = axes[r, c]
        im = ax.imshow(d['rho'].T, origin='lower', cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f'{label}, t = {float(d["t"]) * 1e6:g} $\\mu$s')
        ax.set_xlabel('x [pixels]')
        ax.set_ylabel('y [pixels]')
fig.colorbar(im, ax=axes, label=r'$\rho$', shrink=0.8)
fig.savefig(os.path.join(here, 'figures', fig_name + '_snapshots.png'), dpi=150)

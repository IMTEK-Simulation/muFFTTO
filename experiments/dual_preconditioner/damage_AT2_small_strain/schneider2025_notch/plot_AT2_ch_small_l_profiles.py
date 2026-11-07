"""
Crack profiles of the CH small-l study (run_AT2_cahn_hilliard_small_l.sh): AT2, 128^2, l/h = 4, 2, 1, 0.5, 0.25.
Top row: damage map of the current (latest checkpoint) state, exact pixels, phase-2 outline in cyan.
Bottom: damage across the crack. For every column x with max d > 0.9 the damage is taken along y around its maximum
(load yy -> cracks run roughly along x) and averaged over these columns; markers = nodal values, dashed = the 1D AT2
profile exp(-|y| / l).   python plot_AT2_ch_small_l_profiles.py   ->  figures/AT2_ch_small_l_profiles.png
"""
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
N = 128
RUNS = [(0.03125, '0.03125'), (0.015625, '0.015625'), (0.0078125, '0.0078125'), (0.00390625, '0.00390625'),
        (0.001953125, '0.00195312')]
P = os.environ.get('PRECOND', 'dual')
plt.rcParams.update({'font.size': 13})

geo = np.load(os.path.join(HERE, 'geometry', 'cahn_hilliard_512.npy'))
phase = geo[::geo.shape[0] // N, ::geo.shape[1] // N]

fig, axes = plt.subplots(2, len(RUNS), figsize=(4.2 * len(RUNS), 8.4), layout='constrained',
                         gridspec_kw=dict(height_ratios=[1, 0.8]))
half = 8
offs = np.arange(-half, half + 1)
for k, (l, tag) in enumerate(RUNS):
    c = np.load(os.path.join(HERE, 'exp_data', f'AT2_N{N}_{P}_incl0.1_l{tag}_k0.0001_ch_checkpoint.npz'))
    d = np.asarray(c['damage']).reshape(N, N)            # [x, y]
    lh = l * N
    a = axes[0, k]
    im = a.imshow(d.T, origin='lower', cmap='inferno', vmin=0, vmax=1, interpolation='nearest', extent=(0, N, 0, N))
    a.contour(np.arange(N) + 0.5, np.arange(N) + 0.5, phase.T, levels=[0.5], colors='cyan', linewidths=0.5)
    a.set_title(f'l/h = {lh:g}  (step {int(c["step"])}, max d {d.max():.2f})')
    a.set_xlabel('x [pixels]')
    if k == 0:
        a.set_ylabel('y [pixels]')
    # profile across the crack
    cols = [x for x in range(N) if d[x].max() > 0.9]
    prof = np.array([d[x, (np.argmax(d[x]) + offs) % N] for x in cols]) if cols else np.zeros((1, offs.size))
    b = axes[1, k]
    b.plot(offs, prof.mean(axis=0), 'o-', color='tab:red', lw=2, ms=6, label=f'mean over {len(cols)} columns')
    b.fill_between(offs, prof.min(axis=0), prof.max(axis=0), color='tab:red', alpha=0.15, label='min-max')
    yy = np.linspace(-half, half, 801)
    b.plot(yy, np.exp(-np.abs(yy) / lh), 'k--', lw=1.5, label='exp(-|y|/l)')
    b.set_xlim(-half, half), b.set_ylim(0, 1.12)
    b.set_xlabel('distance from crack centre [pixels]')
    if k == 0:
        b.set_ylabel('damage d')
    b.grid(alpha=0.3)
    b.legend(fontsize=10, loc='upper right')
fig.colorbar(im, ax=axes[0, :], shrink=0.8, label='damage d')
fig.suptitle(f'CH two-phase AT2, 128$^2$, k$_{{res}}$ = 1e-4 ({P}): crack profiles from resolved to sub-pixel l',
             fontsize=15)
out = os.path.join(HERE, 'figures', 'AT2_ch_small_l_profiles.png')
fig.savefig(out, dpi=120)
print('wrote', out)

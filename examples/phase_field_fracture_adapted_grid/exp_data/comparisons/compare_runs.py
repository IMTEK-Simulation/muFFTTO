"""Damage of several runs of the same problem (regular or adapted grid, any resolution), matched by F11.

    python compare_runs.py out.png run_a.nc run_b.nc ... [--F=0.92,0.89,...]   static figure
    python compare_runs.py out.mp4 run_a.nc run_b.nc ...                     movie

Static: top row the P11 response of all runs, then one row per run with the damage on the
deformed cell at the requested F11 values (nearest frame of each run).
Movie: one frame per frame of the finest run; the other runs show their frame nearest in F11.
"""
import sys
sys.path.insert(0, '/home/martin/Programming/microTopOpt/muFFTTO')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from muFFTTO import io_utils

args = [a for a in sys.argv[1:] if not a.startswith('--')]
out, files = args[0], args[1:]
F_list = [0.95, 0.92, 0.89, 0.875, 0.86, 0.845]
for a in sys.argv[1:]:
    if a.startswith('--F'):
        F_list = [float(v) for v in a.split('=', 1)[1].split(',')]
runs = [io_utils.read_file(f) for f in files]
medium = ListedColormap([(0.75, 0.87, 1.0)])
colors = ['C0', 'C3', 'C2', 'C1', 'k', 'C4']


def F11_of(r):
    H = np.asarray(r.attributes['H_macro']).reshape(2, 2)
    return 1.0 + r.frame_variables['lam'] * H[1, 1]


def name(r):
    kind = 'adapted' if int(r.attributes.get('adapted_grid', 0)) else 'regular'
    return f"{kind} {r.fields['phase'].shape[-2]}$^2$"


def cell(ax, r, fr):
    H = np.asarray(r.attributes['H_macro']).reshape(2, 2)
    lam = r.frame_variables['lam'][fr]
    size = np.asarray(r.attributes['domain_size'], float)
    phase = r.fields['phase'][0, 0, 0]
    nx, ny = phase.shape
    i, j = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing='ij')
    X = np.array([i * size[0] / nx, j * size[1] / ny])
    if 'grid_displacement' in r.fields:                    # adapted grid: X + d_g
        X = X + r.fields['grid_displacement'][0][:, 0][:, i % nx, j % ny]
    u = r.fields['u_fluc'][fr][:, 0][:, i % nx, j % ny]
    x, y = np.einsum('ab,b...->a...', np.eye(2) + lam * H, X) + u
    d = r.fields['damage'][fr, 0, 0]
    dp = 0.25 * (d + np.roll(d, -1, 0) + np.roll(d, -1, 1) + np.roll(d, -1, (0, 1)))
    solid = phase > 0
    ax.pcolormesh(x, y, np.ma.masked_where(solid, np.ones_like(dp)), cmap=medium, shading='flat')
    im = ax.pcolormesh(x, y, np.ma.masked_where(~solid, dp), cmap='inferno', vmin=0, vmax=1, shading='flat')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'F11 {F11_of(r)[fr]:.3f}, max d {r.frame_variables["damage_max"][fr]:.2f}', fontsize=8)
    return im


def response(ax, marks=None):
    for k, r in enumerate(runs):
        F, P = F11_of(r), r.frame_variables['mean_P'][:, 1, 1]
        ax.plot(np.r_[1.0, F], np.r_[0.0, P], '.-', ms=2, color=colors[k], label=name(r))
        if marks is not None:
            ax.plot(F[marks[k]], P[marks[k]], 's', color=colors[k], ms=6)
    ax.set_xlabel(r'$\bar F_{11}$')
    ax.set_ylabel(r'$\bar P_{11}$')
    ax.legend(fontsize=8)


if out.endswith('.mp4'):
    from matplotlib import animation
    fine = int(np.argmax([r.nb_frames for r in runs]))   # frames of the run with the most frames
    F_fine = F11_of(runs[fine])
    fig = plt.figure(figsize=(4.2 * (len(runs) + 1), 4.4))

    def draw(fr):
        fig.clf()
        axes = fig.subplots(1, len(runs) + 1)
        frames = [int(np.argmin(np.abs(F11_of(r) - F_fine[fr]))) for r in runs]
        response(axes[0], frames)
        for ax, r, f in zip(axes[1:], runs, frames):
            im = cell(ax, r, f)
            ax.set_title(name(r) + '\n' + ax.get_title(), fontsize=8)
        fig.colorbar(im, ax=list(axes[1:]), label='damage', shrink=0.8)

    movie = animation.FuncAnimation(fig, draw, frames=len(F_fine), interval=100, repeat=False)
    movie.save(out, writer=animation.FFMpegWriter(fps=8), dpi=100)
else:
    fig = plt.figure(figsize=(2.6 * len(F_list), 3.0 + 2.5 * len(runs)), layout='constrained')
    gs = fig.add_gridspec(1 + len(runs), len(F_list), height_ratios=[1.2] + [1] * len(runs))
    ax_r = fig.add_subplot(gs[0, :])
    response(ax_r, [[int(np.argmin(np.abs(F11_of(r) - F))) for F in F_list] for r in runs])
    for F in F_list:
        ax_r.axvline(F, color='0.8', lw=0.5, zorder=0)
    for k, r in enumerate(runs):
        for c, F in enumerate(F_list):
            ax = fig.add_subplot(gs[1 + k, c])
            im = cell(ax, r, int(np.argmin(np.abs(F11_of(r) - F))))
            if c == 0:
                ax.set_ylabel(name(r), fontsize=9)
    fig.colorbar(im, ax=fig.axes[1:], label='damage', shrink=0.6)
    fig.savefig(out, dpi=110)
print('wrote', out)

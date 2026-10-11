"""Side-by-side: damage on the whole cell vs solid only. python damage_domain.py out.png all.nc solid.nc"""
import sys
sys.path.insert(0, '/home/martin/Programming/microTopOpt/muFFTTO')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from muFFTTO import io_utils

out, files = sys.argv[1], sys.argv[2:]
labels = ['damage everywhere (old)', 'damage in solid only (new)']
runs = [io_utils.read_file(f) for f in files]
medium = ListedColormap([(0.75, 0.87, 1.0)])


def cell(ax, r, fr, title):
    H = np.asarray(r.attributes['H_macro']).reshape(2, 2)
    lam = r.frame_variables['lam'][fr]
    size = np.asarray(r.attributes['domain_size'], float)
    phase = r.fields['phase'][0, 0, 0]
    nx, ny = phase.shape
    i, j = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing='ij')
    X = np.array([i * size[0] / nx, j * size[1] / ny])
    u = r.fields['u_fluc'][fr][:, 0][:, i % nx, j % ny]
    x, y = np.einsum('ab,b...->a...', np.eye(2) + lam * H, X) + u
    d = r.fields['damage'][fr, 0, 0]
    dp = 0.25 * (d + np.roll(d, -1, 0) + np.roll(d, -1, 1) + np.roll(d, -1, (0, 1)))
    # show d on ALL pixels, medium hatched light: to see the leak into the medium
    ax.pcolormesh(x, y, np.ma.masked_where(phase > 0, dp), cmap='Blues', vmin=0, vmax=1, shading='flat', alpha=0.9)
    im = ax.pcolormesh(x, y, np.ma.masked_where(phase == 0, dp), cmap='inferno', vmin=0, vmax=1, shading='flat')
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    return im


fig = plt.figure(figsize=(16, 7.5))
gs = fig.add_gridspec(2, 5, width_ratios=[1.6, 1, 1, 1, 0.05])
ax = fig.add_subplot(gs[:, 0])
for r, lab, c in zip(runs, labels, ['C1', 'C3']):
    h = r.frame_variables
    F = 1 + h['lam'] * np.asarray(r.attributes['H_macro']).reshape(2, 2)[1, 1]
    P = h['mean_P'][:, 1, 1]
    ax.plot(np.r_[1, F], np.r_[0, P], '.-', c=c, ms=3, label=lab)
    k = int(np.argmax(np.abs(P)))
    drop = int(np.where(np.diff(P) > 0.15)[0][0]) + 1
    frames = [k, drop, r.nb_frames - 1]
    ax.plot(F[frames], P[frames], 's', c=c, mfc='none', ms=8)
    row = runs.index(r)
    for col, (fr, name) in enumerate(zip(frames, ['peak', 'after 1st crack', 'final'])):
        im = cell(fig.add_subplot(gs[row, col + 1]), r, fr, f'{lab}\n{name}: F11 = {F[fr]:.3f}, P11 = {P[fr]:.3f}')
ax.invert_xaxis(); ax.set_xlabel(r'$\bar F_{11}$'); ax.set_ylabel(r'$\bar P_{11}$'); ax.grid(alpha=0.3); ax.legend()
ax.set_title('sinusoidal_ligaments 32x32, compression, Gc = 0.1, l = 2 px', fontsize=10)
fig.colorbar(im, cax=fig.add_subplot(gs[:, 4]), label='damage (solid: inferno, medium: blue)')
fig.tight_layout()
fig.savefig(out, dpi=90)
print('saved', out)

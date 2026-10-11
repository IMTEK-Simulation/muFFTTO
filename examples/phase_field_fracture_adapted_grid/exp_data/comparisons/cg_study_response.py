"""Response and deformation of the CG-study cases.

    python cg_study_response.py out.png case_folder_list.txt [--F=0.95,0.9,...]

case_folder_list.txt: lines "<case> <run folder>", case = <grid>_<physics>_<preconditioner> (as cg_study.py).
Top: P11 over F11 of all cases, one panel per preconditioner (the physics must not depend on it).
Below: one row per grid + physics (the run that got furthest of the two preconditioners), damage on the
deformed cell at the requested F11 values (nearest frame), the last column the final frame.
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
out, listing = args[0], args[1]
F_list = [0.95, 0.90, 0.85, 0.80, 0.75]
for a in sys.argv[1:]:
    if a.startswith('--F'):
        F_list = [float(v) for v in a.split('=', 1)[1].split(',')]

physics_order = ['elastic', 'contact', 'damage', 'damage_contact']
labels = {'elastic': 'elasticity', 'contact': '+ contact (no damage)', 'damage': '+ damage (no contact)',
          'damage_contact': '+ damage + contact'}
colors = {'elastic': 'C0', 'contact': 'C2', 'damage': 'C1', 'damage_contact': 'C3'}
medium = ListedColormap([(0.75, 0.87, 1.0)])

cases = {}
for line in open(listing):
    if line.strip():
        name, folder = line.split()
        prec = 'Green_Jacobi' if name.endswith('Green_Jacobi') else 'Green'
        grid = name.split('_')[0]
        phys = name[len(grid) + 1:-len(prec) - 1]
        cases[(grid, phys, prec)] = io_utils.read_file(folder + '/contact_fracture.nc')


def F11_of(r):
    H = np.asarray(r.attributes['H_macro']).reshape(2, 2)
    return 1.0 + r.frame_variables['lam'] * H[1, 1]


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
    ax.pcolormesh(x, y, np.ma.masked_where(solid, np.ones_like(dp)), cmap=medium, shading='flat',
                  edgecolors='0.6', linewidth=0.1)
    im = ax.pcolormesh(x, y, np.ma.masked_where(~solid, dp), cmap='inferno', vmin=0, vmax=1, shading='flat')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'F11 {F11_of(r)[fr]:.3f}, P11 {r.frame_variables["mean_P"][fr, 1, 1]:+.3f}', fontsize=8)
    return im


rows = [(g, p) for g in ['regular', 'adapted'] for p in physics_order
        if any((g, p, pr) in cases for pr in ['Green', 'Green_Jacobi'])]
ncol = len(F_list) + 1
fig = plt.figure(figsize=(2.5 * ncol, 4.0 + 2.4 * len(rows)), layout='constrained')
gs = fig.add_gridspec(1 + len(rows), ncol, height_ratios=[1.6] + [1] * len(rows))
for k, prec in enumerate(['Green', 'Green_Jacobi']):
    ax = fig.add_subplot(gs[0, 3 * k:3 * k + 3])
    for (g, p, pr), r in cases.items():
        if pr != prec:
            continue
        ax.plot(np.r_[1.0, F11_of(r)], np.r_[0.0, r.frame_variables['mean_P'][:, 1, 1]], '.-', ms=2, lw=1,
                color=colors[p], ls='-' if g == 'adapted' else '--', label=f'{g}, {labels[p]}')
    ax.set_title(f'preconditioner {prec}', fontsize=10)
    ax.set_xlabel(r'$\bar F_{11}$')
    ax.set_ylabel(r'$\bar P_{11}$')
    ax.set_xlim(0.72, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
for k, (g, p) in enumerate(rows):
    r = min((cases[(g, p, pr)] for pr in ['Green', 'Green_Jacobi'] if (g, p, pr) in cases),
            key=lambda r: F11_of(r)[-1])                 # the run that got furthest
    F = F11_of(r)
    frames = [int(np.argmin(np.abs(F - Fc))) for Fc in F_list if Fc >= F[-1] - 0.01] + [len(F) - 1]
    for c, fr in enumerate(frames):
        ax = fig.add_subplot(gs[1 + k, c])
        im = cell(ax, r, fr)
        if c == 0:
            ax.set_ylabel(f'{g}\n{labels[p]}', fontsize=9)
        if c == len(frames) - 1:
            ax.set_title('final: ' + ax.get_title(), fontsize=8)
fig.colorbar(im, ax=fig.axes[2:], label='damage (solid); blue: void', shrink=0.4)
fig.savefig(out, dpi=100)
print('wrote', out)

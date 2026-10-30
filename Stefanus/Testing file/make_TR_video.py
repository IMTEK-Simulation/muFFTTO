"""
Video of a stefanus_2D_finite_strain_NeoHookean_TR.py run (--frames): the deformed periodic medium (2 x 2 cells,
matrix coloured by the axial stress P_kk, voids white, stiff inclusions blue) and the nominal stress-strain curve.

usage:  python make_TR_video.py X 32x28 [32x56_cells1x2 ...]   -> figures/finite_strain_TR_X_32x28.mp4 (+ _last.png)
        further runs (e.g. a supercell) are drawn as extra curves; the video shows the first run
"""
import glob
import os
import re
import subprocess
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
direction, grid, others = sys.argv[1], sys.argv[2], sys.argv[3:]
cells = re.search(r'cells(\d+x\d+)', grid)[1] if 'cells' in grid else '1x1'


def describe(g):
    """readable legend label of a frame-folder tag, e.g. '64 px, contrast 10, Lambda/mu 100, third medium k = 1e-3'"""
    parts = [g.split('x')[0] + ' px']
    m = re.search(r'_c(\d+)_L(\d+)', g)
    parts.append(f'contrast {m[1]}, $\\Lambda/\\mu$ {m[2]}' if m else 'paper material')
    tm = re.search(r'_TM([\d.e-]+?)_', g + '_')
    parts.append(f'third medium k = {float(tm[1]):g}' if tm else 'true holes')
    return ', '.join(parts)


def curve(g):
    fs = sorted(glob.glob(os.path.join(HERE, 'exp_data', f'frames_TR_{direction}_{g}', 'inc*.npz')))
    return (np.r_[0.0, [float(np.load(f)['eps']) for f in fs]], np.r_[0.0, [float(np.load(f)['stress_MPa']) for f in fs]])


other_curves = [(g, *curve(g)) for g in others]
frames = sorted(glob.glob(os.path.join(HERE, 'exp_data', f'frames_TR_{direction}_{grid}', 'inc*.npz')))
data = [dict(np.load(f)) for f in frames]
eps = np.array([0.0] + [float(d['eps']) for d in data])
stress = np.array([0.0] + [float(d['stress_MPa']) for d in data])
Lx, Ly = data[0]['domain_size']
mm = float(data[0]['R_MM'])
phase = data[0]['phase']
nx, ny = phase.shape
X, Y = np.meshgrid(np.arange(nx + 1) * Lx / nx, np.arange(ny + 1) * Ly / ny, indexing='ij')
# colour range: 99th percentile of |P_kk| in the matrix over all frames (a few stress peaks would wash out the rest)
vmax = np.percentile(np.abs(np.concatenate([d['P_kk'][phase == 0].ravel() for d in data])), 99)
norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
paper = {'X': 0.35, 'Y': 0.50}[direction]
fig4 = np.genfromtxt(os.path.join(HERE, 'exp_data', 'paper_fig4_compositeA.csv'), delimiter=',', skip_header=2,
                     names=True, dtype=None, encoding=None)
fig4 = {c: (fig4['eps'][(fig4['direction'] == direction) & (fig4['curve'] == c)],
            fig4['stress'][(fig4['direction'] == direction) & (fig4['curve'] == c)]) for c in ('periodic', 'experiment')}
eps_cr = {'X': 0.016, 'Y': 0.026}[direction]
out_dir = os.path.join(HERE, 'figures', f'video_TR_{direction}_{grid}')
os.makedirs(out_dir, exist_ok=True)
plt.rcParams.update({'font.size': 12})

# fixed view: the deformed 2 x 2 cells of the last frame and the undeformed ones
Fl = data[-1]['F_macro']
x_max = max(2 * Lx, (Fl @ [2 * Lx, 2 * Ly])[0]) * mm
y_max = max(2 * Ly, (Fl @ [2 * Lx, 2 * Ly])[1]) * mm

for i, d in enumerate(data):
    Fm = d['F_macro']
    ux = np.pad(d['u'][0], ((0, 1), (0, 1)), mode='wrap')
    uy = np.pad(d['u'][1], ((0, 1), (0, 1)), mode='wrap')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), layout='constrained', gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axes[0]
    matrix_stress = np.ma.masked_where(phase != 0, d['P_kk'])
    for a in range(2):
        for b in range(2):
            Xs, Ys = X + a * Lx, Y + b * Ly
            xd = (Fm[0, 0] * Xs + Fm[0, 1] * Ys + ux) * mm
            yd = (Fm[1, 0] * Xs + Fm[1, 1] * Ys + uy) * mm
            pcm = ax.pcolormesh(xd, yd, matrix_stress, cmap='coolwarm', norm=norm, shading='flat')
            ax.pcolormesh(xd, yd, np.ma.masked_where(phase != 2, phase), cmap=matplotlib.colors.ListedColormap(
                ['#4d4d4d']), shading='flat')
    ax.plot(np.array([0, 2, 2, 0, 0]) * Lx * mm, np.array([0, 0, 2, 2, 0]) * Ly * mm, 'k--', lw=0.8)
    # the computed (simulation) cell: the periodic box deformed by F_macro (the lattice that the periodic copies repeat;
    # its edges cut through voids, where a true-hole run has no meaningful displacement)
    box = Fm @ (np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]]) * np.array([[Lx], [Ly]]))
    ax.plot(box[0] * mm, box[1] * mm, 'k-', lw=2)
    ax.set(aspect='equal', xlim=(-0.5, x_max + 0.5), ylim=(-0.5, y_max + 0.5), xlabel='x [mm]', ylabel='y [mm]',
           title=f'compression in {direction}: $\\varepsilon$ = {float(d["eps"]):.3f}\n'
                 f'solid: computed {cells} cell, repeated 2 x 2; dashed: undeformed')
    fig.colorbar(pcm, ax=ax, shrink=0.8, extend='both',
                 label=f'axial stress $P_{{{direction.lower() * 2}}}$ in the matrix [MPa]')
    ax = axes[1]
    ax.plot(eps, stress, '-', color='0.75', lw=1.5)
    for (c, e_o, s_o), col, ls in zip(other_curves, ['tab:purple', 'tab:orange', 'tab:brown', 'tab:gray', 'tab:olive'],
                                      ['-', '--', ':', '-', '--']):
        ax.plot(e_o, s_o, ls, color=col, lw=2, label=describe(c))
    ax.plot(eps[:i + 2], stress[:i + 2], 'o-', color='tab:green', ms=4, lw=2, label=f'{describe(grid)} (shown left)')
    ax.plot(*fig4['periodic'], color='tab:red', ls='-.', lw=1.8, label='paper Fig. 4: simulation, periodic')
    ax.plot(*fig4['experiment'], color='k', ls='-', lw=1.5, label='paper Fig. 4: experiment')
    ax.axvline(eps_cr, color='tab:red', ls=':', lw=1.2)
    ax.annotate(f'paper $\\varepsilon_{{cr}}$ = {eps_cr}\n(buckling, 1 x 2 cells)', (eps_cr, 0.92 * stress.max()),
                xytext=(6, 0), textcoords='offset points', color='tab:red', fontsize=10.5)
    ax.set(xlabel=f'nominal strain $\\varepsilon_{direction}$', ylabel='nominal stress [MPa]',
           xlim=(0, 0.205), ylim=(0, max([stress.max(), 0.016] + [c[2].max() for c in other_curves]) * 1.08),
           title='nominal stress vs strain (lateral side stress free)')
    ax.legend(frameon=False, loc='upper left')
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(out_dir, f'frame{i:04d}.png'), dpi=100)
    plt.close(fig)

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG = 'ffmpeg'
video = os.path.join(HERE, 'figures', f'finite_strain_TR_{direction}_{grid}.mp4')
subprocess.run([FFMPEG, '-y', '-loglevel', 'error', '-framerate', '6', '-i', os.path.join(out_dir, 'frame%04d.png'),
                '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p', video], check=True)
subprocess.run(['cp', os.path.join(out_dir, f'frame{len(data) - 1:04d}.png'),
                video.replace('.mp4', '_last.png')], check=True)
print(video)

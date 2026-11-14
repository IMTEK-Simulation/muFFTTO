"""3D fatigue crack of the NEMO 128^3 run (Green; the solution is identical for all preconditioners), see README.md.

    python plot_FAT_3D_crack.py [N] [PREC] [still|video|both]
    -> figures/FAT_3D_crack_N<N>.png / .pdf  (still: pre-crack, mid growth, last stored state before the final break)
       figures/FAT_3D_crack_N<N>.mp4 (+ _final.png): one stored snapshot (every 5th cycle) per 0.5 s, slow rotation

Damage isosurface d = 0.9 (skimage marching_cubes on the nodal damage, one value per voxel), glass beads (phase < 0.5)
semi-transparent. The loading direction y is drawn vertical, so the pre-crack (plane y = L/2, through z) is horizontal.
"""
import os
import sys

import imageio_ffmpeg
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

HERE = os.path.dirname(os.path.realpath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
PREC = sys.argv[2] if len(sys.argv) > 2 else 'Green'
MODE = sys.argv[3] if len(sys.argv) > 3 else 'both'
TAG = '_incl11.6667_l0.015625_3D_k0.0001_nemo3D_precrack_asymptotic_amp0.0022_cyc1500'
E_MPA = 6000.0
LEVEL = 0.9
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})

base = os.path.join(HERE, 'exp_data', f'FAT_N{N}_{PREC}{TAG}')
nc = netCDF4.Dataset(base + '.nc')
steps = list(np.asarray(nc['step'][:]).astype(int))
sig = list(np.asarray(nc['macro_stress'][:, 1, 1]) * E_MPA)
fin = np.load(base + '_damage.npz')
cyc = np.genfromtxt(base + '_cycles.csv', delimiter=',', names=True)
get = [lambda k=k: np.asarray(nc['damage'][k, 0]) for k in range(len(steps))]
if int(fin['steps'][-1]) > steps[-1]:
    steps.append(int(fin['steps'][-1]))
    sig.append(cyc['sigma'][int(fin['steps'][-1])] * E_MPA)
    get.append(lambda: np.asarray(fin['damage'][-1]))
LIGHT = np.array([0.4, 0.77, 0.5])


def to_plot(v):
    """index coords (x, y, z) -> plot coords (x, z, y): load direction y vertical"""
    return v[:, [0, 2, 1]]


def shaded(v, f, cmap, lo, hi):
    nrm = np.cross(v[f][:, 1] - v[f][:, 0], v[f][:, 2] - v[f][:, 0])
    s = np.abs(nrm @ LIGHT) / (np.linalg.norm(nrm, axis=1) * np.linalg.norm(LIGHT) + 1e-30)
    return cmap(lo + (hi - lo) * s)


def iso(field, level, step=1):
    a = np.pad(field, 1, mode='constant', constant_values=0.0)          # closed at the cell boundary
    v, f, _, _ = measure.marching_cubes(a, level, step_size=step)
    return to_plot((v - 1 + 0.5) / N), f                                # voxel centres at (i + 1/2) h


bv, bf = iso((fin['phase'] < 0.5).astype(float), 0.5, step=max(1, N // 64))
bead_col = shaded(bv, bf, plt.cm.Blues, 0.25, 0.55)
bead_col[:, 3] = 0.18


def draw(ax, dmg, title, azim=-60, elev=18):
    ax.cla()
    ax.add_collection3d(Poly3DCollection(bv[bf], facecolors=bead_col, linewidth=0))
    if dmg.max() > LEVEL:
        v, f = iso(dmg, LEVEL)
        ax.add_collection3d(Poly3DCollection(v[f], facecolors=shaded(v, f, plt.cm.YlOrRd, 0.45, 0.95),
                                             linewidth=0))
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0], color='0.5', lw=0.6)
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [1, 1, 1, 1, 1], color='0.5', lw=0.6)
    for x, y in ((0, 0), (1, 0), (1, 1), (0, 1)):
        ax.plot([x, x], [y, y], [0, 1], color='0.5', lw=0.6)
    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), xticks=[], yticks=[], zticks=[])
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=13, y=0.97)


if MODE in ('still', 'both'):
    k_pre = len(steps) - 2 if len(steps) > 1 else 0                       # last stored state before the break
    sel = (0, k_pre // 2 + 2, k_pre)                                      # pre-crack, mid growth, before the break
    fig = plt.figure(figsize=(12.6, 4.6))
    for i, k in enumerate(sel):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d', computed_zorder=False)
        draw(ax, get[k](), f'cycle {steps[k] + 1}, peak stress {sig[k]:.1f} MPa')
    fig.text(0.5, 0.03, f'${N}^3$ ($\\ell/h = {N // 64}$): damage $d = {LEVEL}$ (red), glass beads (blue), '
             'load vertical', ha='center', fontsize=12)
    fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0.07, wspace=0)
    out = os.path.join(HERE, 'figures', f'FAT_3D_crack_N{N}')
    fig.savefig(out + '.png', dpi=300)
    fig.savefig(out + '.pdf')
    print('wrote', out + '.png/.pdf')
    plt.close(fig)

if MODE in ('video', 'both'):
    PER, FPS, HOLD = 6, 12, 18
    W, H, DPI = 2400, 2160, 200
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0.02, 1, 0.9], projection='3d', computed_zorder=False)
    head = fig.text(0.5, 0.955, '', ha='center', va='center', fontsize=16)
    fig.text(0.5, 0.03, f'${N}^3$ ($\\ell/h = {N // 64}$): damage $d = {LEVEL}$ (red), glass beads (blue), load vertical', ha='center',
             fontsize=13)
    out = os.path.join(HERE, 'figures', f'FAT_3D_crack_N{N}.mp4')
    w = imageio_ffmpeg.write_frames(out, (W, H), fps=FPS, quality=9, codec='libx264', pix_fmt_out='yuv420p',
                                    macro_block_size=8)
    w.send(None)
    nfr = len(steps) * PER
    for k in range(len(steps)):
        dmg = get[k]()
        for j in range(PER):
            i = k * PER + j
            draw(ax, dmg, '', azim=-75 + 50 * i / nfr)
            head.set_text(f'cycle {steps[k] + 1}, peak stress {sig[k]:.1f} MPa')
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            w.send(buf)
    for _ in range(HOLD):
        w.send(buf)
    w.close()
    fig.savefig(out.replace('.mp4', '_final.png'), dpi=DPI)
    print('saved', out)

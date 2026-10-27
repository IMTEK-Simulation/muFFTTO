"""
Video of the cantilever crack mesh study (cantilever_crack_precond.py): the mesh is refined n_H = 16 -> 256 and on
every mesh the tip load is ramped up, so the beam bends and the crack near the clamp opens.

    left    the deformed beam (displacement magnified, the same factor on every mesh of a case), every pixel coloured
            by the damage d at its centre (sand = intact, dark red = broken); the undeformed outline in grey
    right   CG iterations vs n_H of the four preconditioners, up to the current mesh

rows: mild crack (d = exp(-dist / l)) and broken core (d = 1 within 0.5 l), each with l = 2h and l = 0.02 fixed.
Fields from  FIELDS=1 PRECONDS=green_c python cantilever_crack_precond.py  (exp_data/runs_crack_fields/),
iterations from exp_data/runs_crack/.

usage:  python crack_deformation_video.py [--workers 4]      -> exp_data/cantilever_crack_deformation.mp4
"""
import argparse
import os
import shutil
import subprocess
from multiprocessing import Pool

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import NullFormatter

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=4)
args = parser.parse_args()

HERE = os.path.dirname(os.path.realpath(__file__))
RUNS = os.path.join(HERE, 'exp_data', 'runs_crack')
FIELDS = os.path.join(HERE, 'exp_data', 'runs_crack_fields')
FRAMES = os.path.join(HERE, 'exp_data', 'frames_crack_deformation')
L, H = 1.0, 0.2
MESHES = [16, 32, 64, 128, 256]
RAMP, HOLD = 12, 8  # frames per mesh: load ramp 0 -> 1, then hold
TIP = 0.25  # magnified tip deflection, in units of L
CASES = [('l2h_k0.0001_eps0.001', r'mild crack, $\ell = 2h$'),
         ('l0.02_k0.0001_eps0.001', r'mild crack, $\ell = 0.02$ fixed'),
         ('l2h_k0.0001_eps0.001_core0.5', r'broken core, $\ell = 2h$'),
         ('l0.02_k0.0001_eps0.001_core0.5', r'broken core, $\ell = 0.02$ fixed')]
LABELS = {'green': 'Green, projected', 'green_c': 'Green, corrected', 'dual': 'dual, projected',
          'dual_c': 'dual, corrected'}
STYLES = {'green': dict(color='#86b6ef', ls=(0, (5, 3)), marker='s', mfc='white'),
          'green_c': dict(color='#184f95', ls='-', marker='o'),
          'dual': dict(color='#f4a582', ls=(0, (5, 3)), marker='D', mfc='white'),
          'dual_c': dict(color='#c4441a', ls='-', marker='^')}
CMAP = colors.LinearSegmentedColormap.from_list(
    'damage', [(0, '#d9c7a3'), (0.25, '#d9c7a3'), (0.5, '#e08a3c'), (0.75, '#b2182b'), (1, '#3d0008')])


def field(t, n):
    f = os.path.join(FIELDS, f'{t}_green_c_n{n:04d}.npz')
    return np.load(f) if os.path.exists(f) else None


def iterations(t):
    its = {}
    for p in LABELS:
        rows = [(n, int(np.load(f)['cg_its'])) for n in MESHES
                if os.path.exists(f := os.path.join(RUNS, f'{t}_{p}_n{n:04d}.npz'))]
        if rows:
            its[p] = np.array(rows)
    return its


def magnification(t):
    """the same factor on every mesh of a case: the finest available mesh bends the tip by TIP"""
    for n in MESHES[::-1]:
        r = field(t, n)
        if r is not None:
            return TIP / np.abs(r['u'][1][r['beam']]).max()


def frame_list():
    meshes = [n for n in MESHES if all(field(t, n) is not None for t, _ in CASES)]
    return [(n, min(k / (RAMP - 1), 1.0)) for n in meshes for k in range(RAMP + HOLD)]


def render(jobs):
    mags = {t: magnification(t) for t, _ in CASES}
    its = {t: iterations(t) for t, _ in CASES}
    fig = plt.figure(figsize=(16, 13))
    for i, (n, load) in jobs:
        fig.clf()
        gs = fig.add_gridspec(4, 2, height_ratios=(1.5, 1, 1.5, 1), left=0.05, right=0.98, top=0.93, bottom=0.1,
                              hspace=0.3, wspace=0.12)
        for row, (t, title) in enumerate(CASES):
            r = field(t, n)
            nv, h = int(r['nv']), float(r['h'])
            beam = r['beam']
            nx, ny = beam.shape
            u = np.pad(r['u'], ((0, 0), (0, 1), (0, 1)), mode='wrap') * load * mags[t]  # node (i, j) of pixel (i, j)
            gx, gy = np.meshgrid((np.arange(nx + 1) - nv) * h, (np.arange(ny + 1) - nv) * h, indexing='ij')
            ax = fig.add_subplot(gs[2 * (row // 2), row % 2])  # mild top, broken core bottom
            ax.plot([0, L, L, 0, 0], [0, 0, H, H, 0], color='0.6', lw=0.8, ls=(0, (4, 3)))
            ax.pcolormesh(gx + u[0], gy + u[1], np.ma.masked_where(~beam, r['d']), cmap=CMAP, vmin=0, vmax=1,
                          shading='flat', edgecolors='none', rasterized=True)
            ax.plot([-0.012, -0.012], [-0.03, H + 0.03], color='k', lw=3)  # clamp
            ax.set_xlim(-0.03, 1.05)
            ax.set_ylim(-1.08 * TIP - 0.02, H + 0.03)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f'{title}:  $n_H$ = {n},  max d = {float(r["d_max"]):.3f},  load {load:.0%},  '
                         f'displacement × {mags[t]:.0f}', fontsize=10, loc='left')
            ax = fig.add_subplot(gs[2 * (row // 2) + 1, row % 2])
            for p, a in its[t].items():
                ax.plot(a[:, 0], a[:, 1], **STYLES[p], lw=1, ms=4, alpha=0.2)
                now = a[a[:, 0] <= n]
                ax.plot(now[:, 0], now[:, 1], **STYLES[p], lw=2, ms=7, mew=1.5,
                        label=LABELS[p] if row == 0 else None)
            ax.axvline(n, color='0.5', lw=1, zorder=0)
            ax.set_xscale('log', base=2)
            ax.set_xticks(MESHES, [str(m) for m in MESHES])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_ylim(bottom=0)
            ax.set_ylabel('CG iterations')
            ax.grid(True, color='0.9')
            if row >= 2:
                ax.set_xlabel('$n_H$ (pixels across the beam height)')
        fig.legend(loc='lower right', bbox_to_anchor=(0.99, 0.0), ncol=2, frameon=False, fontsize=11, handlelength=4)
        fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap=CMAP),
                     cax=fig.add_axes((0.12, 0.04, 0.3, 0.012)), orientation='horizontal', label='damage d')
        fig.suptitle('clamped cantilever with a frozen phase-field crack at x = 0.1: deformation under mesh '
                     'refinement (corrected Green solution) and CG iterations to $10^{-6}$', fontsize=13)
        fig.savefig(os.path.join(FRAMES, f'g{i:04d}.png'), dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    frames = list(enumerate(frame_list()))
    shutil.rmtree(FRAMES, ignore_errors=True)
    os.makedirs(FRAMES)
    w = min(args.workers, len(frames))
    with Pool(w) as pool:
        pool.map(render, [frames[k::w] for k in range(w)])
    import imageio_ffmpeg
    out = os.path.join(HERE, 'exp_data', 'cantilever_crack_deformation.mp4')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-framerate', '10', '-i',
                    os.path.join(FRAMES, 'g%04d.png'), '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p',
                    out], check=True)
    shutil.copy(os.path.join(FRAMES, f'g{len(frames) - 1:04d}.png'), out.replace('.mp4', '_last.png'))
    print(f'wrote {out} ({len(frames)} frames, meshes {sorted({n for n, _ in frame_list()})})')

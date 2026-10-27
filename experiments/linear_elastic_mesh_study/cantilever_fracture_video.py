"""
Video of a cantilever fracture run (cantilever_fracture.py): one frame per staggered iteration.

    top           the deformed beam (true scale), every pixel coloured by its damage d (mean of its 4 nodes) over the
                  heterogeneous stiffness (grey: light = soft, dark = stiff); holes, void and broken pixels (d > 0.9)
                  white, so a crack shows as a gap
    bottom left   tip force vs tip deflection (absolute, E = 1, beam length 1)
    bottom right  CG iterations of every elastic solve along the run, corrected Green and corrected dual (each from
                  its own run on the same mesh; the two runs follow the same path)

usage:  N_H=64 python cantilever_fracture_video.py [--workers 4] [--fields dual_c]
writes exp_data/cantilever_fracture_nH<n>.mp4 and ..._last.png
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

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--fields', default='dual_c', help='the run whose fields are shown')
args = parser.parse_args()

HERE = os.path.dirname(os.path.realpath(__file__))
OUT = os.path.join(HERE, 'exp_data', 'fracture')
n_H = int(os.environ.get('N_H', 32))
TAIL = f'_holes{os.environ.get("NB_HOLES", 4)}_het{os.environ.get("HETERO", 10)}_l{os.environ.get("L_SCALE", 0.0125)}' \
       f'_ft{os.environ.get("F_T", 0.01)}' + ('_amor' if os.environ.get('MODEL', 'amor') == 'amor' else '')
RUNS = {p: os.path.join(OUT, f'nH{n_H}_{p}{TAIL}') for p in ('green_c', 'dual_c')}
FRAMES = os.path.join(HERE, 'exp_data', f'frames_fracture_nH{n_H}{"_amor" if TAIL.endswith("_amor") else ""}')
LABELS = {'green_c': 'Green, corrected', 'dual_c': 'dual, corrected'}
STYLES = {'green_c': dict(color='#184f95', marker='o'), 'dual_c': dict(color='#c4441a', marker='^')}
CMAP = colors.LinearSegmentedColormap.from_list(
    'damage', [(0, '#e08a3c'), (0.5, '#b2182b'), (1, '#3d0008')])


def frame_variables(base):
    import netCDF4
    with netCDF4.Dataset(base + '.nc') as ds:
        return {k: np.asarray(v[:]) for k, v in ds.variables.items() if v.dimensions == ('frame',)}, \
            {k: ds.getncattr(k) for k in ds.ncattrs()}


def render(frames):
    import netCDF4
    ds = netCDF4.Dataset(RUNS[args.fields] + '.nc')
    fv, at = frame_variables(RUNS[args.fields])
    runs = {p: frame_variables(b)[0] for p, b in RUNS.items() if os.path.exists(b + '.nc')}
    solid, s0 = np.load(RUNS[args.fields] + '_solid.npy')
    solid = solid > 0
    h, nv = float(at['h']), int(at['nv'])
    nx, ny = solid.shape
    gx, gy = np.meshgrid((np.arange(nx + 1) - nv) * h, (np.arange(ny + 1) - nv) * h, indexing='ij')
    # intact colour: grey by the stiffness (s_0 in [1/kappa, 1], log scale)
    grey = np.where(solid, 0.55 - 0.35 * np.log(np.maximum(s0, 1e-12)) / np.log(float(at['kappa'])), 1.0)
    tc, fc = float(at['target_crit']), float(at['force_crit'])
    conv = fv['converged'] > 0
    fig = plt.figure(figsize=(16, 9.5))
    for f in frames:
        fig.clf()
        d_n = np.asarray(ds['damage'][f]).reshape(nx, ny)
        d = 0.25 * (d_n + np.roll(d_n, -1, 0) + np.roll(d_n, -1, 1) + np.roll(d_n, (-1, -1), (0, 1)))
        u = np.pad(np.asarray(ds['u'][f]).reshape(2, nx, ny), ((0, 0), (0, 1), (0, 1)), mode='wrap')
        rgba = np.repeat(grey[..., None], 4, axis=-1)
        rgba[..., 3] = 1
        w = np.clip((d - 0.2) / 0.3, 0, 1)[..., None]  # blend into the damage colours from d = 0.2 on
        rgba = (1 - w) * rgba + w * CMAP(np.clip(d, 0, 1))
        rgba[~solid | (d > 0.9)] = (1, 1, 1, 0)  # voids, holes and broken pixels (their displacement is meaningless)
        ax = fig.add_axes((0.04, 0.4, 0.92, 0.52))
        ax.plot([-0.012, -0.012], [-0.03, 0.23], color='k', lw=4)
        ax.pcolormesh(gx + u[0], gy + u[1], rgba, shading='flat', edgecolors='none', rasterized=True)
        ax.set_xlim(-0.03, 1.08)
        ax.set_ylim(-0.03 + min(0.0, 1.05 * fv['target'].min()), 0.23 + max(0.0, 1.05 * fv['target'].max()))
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'$n_H$ = {n_H} (h = {h:g}, $\\ell$ = {float(at["length_scale"]):g} = '
                     f'{float(at["length_scale"]) / h:g} h),  load step {int(fv["step"][f])}, staggered iteration '
                     f'{int(fv["staggered"][f])},  tip deflection {fv["target"][f]:.3f} (true scale),  '
                     f'max d {fv["d_max"][f]:.2f}', fontsize=11, loc='left')
        ax = fig.add_axes((0.06, 0.07, 0.36, 0.28))
        ax.plot(fv['target'][conv], 1e5 * fv['force'][conv], color='0.75', lw=1)
        upto = conv & (np.arange(len(conv)) <= f)
        ax.plot(fv['target'][upto], 1e5 * fv['force'][upto], 'o-', color='0.2', ms=3)
        ax.plot(fv['target'][f], 1e5 * fv['force'][f], 'o', color='#b2182b', ms=9, mfc='w', mew=2)
        ax.set_xlabel('tip deflection (beam length 1)')
        ax.set_ylabel('tip force × $10^5$ (E = 1)')
        ax.set_ylim(bottom=0)
        ax.grid(True, color='0.9')
        ax = fig.add_axes((0.52, 0.07, 0.44, 0.28))
        for p, r in runs.items():
            k = np.arange(len(r['cg_u']))
            ax.plot(k, r['cg_u'], color=STYLES[p]['color'], lw=0.8, alpha=0.2)
            m = k <= f
            ax.plot(k[m], r['cg_u'][m], color=STYLES[p]['color'], lw=1.5,
                    label=f'{LABELS[p]}: {int(r["cg_u"].sum())} in total')
        ax.axvline(f, color='0.5', lw=1)
        ax.set_xlabel('staggered iteration (whole run)')
        ax.set_ylabel('CG iterations per elastic solve')
        ax.set_ylim(bottom=0)
        ax.grid(True, color='0.9')
        ax.legend(frameon=False, fontsize=10, loc='upper left')
        fig.suptitle(f'AT2 fracture of a clamped heterogeneous cantilever with {int(at["nb_holes"])} holes (stiffness '
                     f'contrast {float(at["kappa"]):g}), tip deflection controlled', fontsize=13)
        fig.savefig(os.path.join(FRAMES, f'g{f:04d}.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    fv, _ = frame_variables(RUNS[args.fields])
    nb = len(fv['step'])
    shutil.rmtree(FRAMES, ignore_errors=True)
    os.makedirs(FRAMES)
    w = min(args.workers, nb)
    with Pool(w) as pool:
        pool.map(render, [list(range(nb))[k::w] for k in range(w)])
    import imageio_ffmpeg
    out = os.path.join(HERE, 'exp_data', f'cantilever_fracture_nH{n_H}{TAIL[TAIL.find("_ft") + 1:].replace("ft0.01", "")}.mp4')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-framerate', '12', '-i',
                    os.path.join(FRAMES, 'g%04d.png'), '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p',
                    out], check=True)
    shutil.copy(os.path.join(FRAMES, f'g{nb - 1:04d}.png'), out.replace('.mp4', '_last.png'))
    print(f'wrote {out} ({nb} frames)')

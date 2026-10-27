"""
Video of the cracking house from the NetCDF of house_damage_3D.py: one frame per converged load step (the last
staggered iteration of each step), rendered in parallel.

    left, right   the deformed house (displacement magnified) seen from the windward and the leeward side, masonry
                  coloured by the damage d (sand = intact, dark red = broken); panels blue
    bottom        wind pressure vs. horizontal eaves displacement; CG iterations of the elastic and the damage solve
                  per staggered iteration

usage:  H=0.125 python house_damage_video.py [--every 1] [--magnify 0]     (0: automatic)
writes exp_data/house_damage_h<h>_ft<f_t>[_nograv].mp4 and its last frame as ..._last.png
"""
import argparse
import os
import shutil
import subprocess
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import LightSource

import house_wind_3D as house
import house_damage_3D as damage_run

parser = argparse.ArgumentParser()
parser.add_argument('--every', type=int, default=1)
parser.add_argument('--magnify', type=float, default=0)
parser.add_argument('--workers', type=int, default=12)
parser.add_argument('--stagger', type=int, default=5,
                    help='also show the staggered iterations of load steps that needed more than this many')
args = parser.parse_args()

HERE = os.path.dirname(os.path.realpath(__file__))
FRAMES = os.path.join(HERE, 'exp_data', 'house', f'frames_{damage_run.case}')
NC = os.path.join(house.OUT, f'{damage_run.case}.nc')
CMAP = colors.LinearSegmentedColormap.from_list(  # sand up to d = 0.25 (AT2 leaves a diffuse low damage everywhere)
    'damage', [(0, '#d9c7a3'), (0.25, '#d9c7a3'), (0.5, '#e08a3c'), (0.75, '#b2182b'), (1, '#3d0008')])


def load():
    """lazy: the frame variables (small) and two readers of one frame's damage / displacement each, so that a
    worker never holds the whole run (several GB per field at h = 0.125)"""
    import netCDF4
    ds = netCDF4.Dataset(NC, 'r')
    fv = {name: np.asarray(v[:]) for name, v in ds.variables.items()
          if v.dimensions == ('frame',)}
    return (lambda f: np.asarray(ds.variables['damage'][f, 0]),
            lambda f: np.asarray(ds.variables['u'][f, :, 0]), fv)


def selected_frames(fv):
    """converged frames, plus every staggered iteration of the load steps that needed more than --stagger"""
    step = fv['step'].astype(int)
    nb_per_step = np.bincount(step)
    keep = (fv['converged'] > 0) | (nb_per_step[step] > args.stagger)
    if not fv['converged'][step == step[-1]].any():  # last step cut off at a deadline: drop its partial iterations
        keep &= step != step[-1]
    return np.flatnonzero(keep)


def render(frames):
    d_all, u_all, fv = load()
    sel = selected_frames(fv)
    phase, base = house.geometry()
    ny_top = phase.shape[1]
    keep = slice(base, ny_top)
    ph = phase[:, keep, :]
    solid = (ph > 0).transpose(0, 2, 1)
    shape = ph.shape
    converged = np.flatnonzero(fv['converged'] > 0)
    node_solid = np.zeros(phase.shape, bool)  # nodes of solid voxels (void nodes carry meaningless values)
    for c in np.ndindex(2, 2, 2):
        node_solid |= np.roll(phase > 0, c, (0, 1, 2))
    magnify = args.magnify or 0.5 / max(np.abs(u_all(converged[-1])[:, node_solid]).max(), 1e-12)
    gx, gy, gz = np.meshgrid(*(np.arange(n + 1) * house.h for n in shape), indexing='ij')
    load_kpa = fv['load'] * house.q_wind / 1e3

    fig = plt.figure(figsize=(16, 10))
    for k in frames:
        fig.clf()
        f = sel[k]
        d_nodal = d_all(f)
        # voxel damage: mean of its 8 nodes
        d_vox = sum(np.roll(d_nodal, (-a, -b, -c), (0, 1, 2)) for a, b, c in np.ndindex(2, 2, 2)) / 8
        d_vox = d_vox[:, keep, :]
        u = np.pad(u_all(f), ((0, 0), (0, 1), (0, 1), (0, 1)), mode='wrap')[:, :, base:ny_top + 1, :]
        X, Y, Z = gx + magnify * u[0], gy + magnify * u[1], gz + magnify * u[2]
        fc = CMAP(np.clip(d_vox, 0, 1))
        fc[ph == 2] = colors.to_rgba('#5b9bd5', 0.9)
        for j, (elev, azim, title) in enumerate(((18, -125, 'windward side'), (25, 55, 'leeward side'))):
            ax = fig.add_axes((0.0 + 0.5 * j, 0.34, 0.5, 0.6), projection='3d')
            ax.voxels(X.transpose(0, 2, 1), Z.transpose(0, 2, 1), Y.transpose(0, 2, 1), solid,
                      facecolors=fc.transpose(0, 2, 1, 3), edgecolor=(0, 0, 0, 0.06), linewidth=0.2,
                      lightsource=LightSource(azdeg=azim + 90, altdeg=elev + 25))
            ax.view_init(elev=elev, azim=azim)
            ax.set_box_aspect((shape[0], shape[2], shape[1]))
            ax.set_axis_off()
            ax.set_title(title, fontsize=11)
        ax = fig.add_axes((0.06, 0.06, 0.38, 0.24))
        c = converged[converged <= f]
        ax.plot(np.r_[0, 1e3 * fv['u_eaves'][converged]], np.r_[0, load_kpa[converged]], color='0.75', lw=1)
        ax.plot(np.r_[0, 1e3 * fv['u_eaves'][c]], np.r_[0, load_kpa[c]], 'o-', color='C3', ms=3)
        ax.plot(1e3 * fv['u_eaves'][f], load_kpa[f], 'o', color='C3', ms=9, mfc='w', mew=2)
        ax.set_xlabel('mean horizontal eaves displacement (controlled) [mm]')
        ax.set_ylabel('wind pressure q [kPa]')
        ax.grid(True, color='0.9')
        ax = fig.add_axes((0.56, 0.06, 0.38, 0.24))
        upto = np.arange(f + 1)
        ax.semilogy(upto, fv['cg_u'][:f + 1], color='C0', lw=1, label='elastic CG, 2 solves (corrected Green)')
        ax.semilogy(upto, fv['cg_d'][:f + 1], color='C2', lw=1, label='damage CG (Green-Jacobi)')
        ax.set_xlim(0, len(fv['cg_u']))
        ax.set_ylim(1, max(fv['cg_u'].max(), fv['cg_d'].max()) * 2)
        ax.set_xlabel('staggered iteration (all load steps)')
        ax.set_ylabel('CG iterations')
        ax.legend(frameon=False, fontsize=9, loc='lower right')
        ax.grid(True, color='0.9')
        fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap=CMAP), cax=fig.add_axes((0.47, 0.4, 0.01, 0.45)),
                     label='damage d')
        fig.suptitle(f'house, AT2 fracture, voxel {house.h * 100:g} cm' + ('' if damage_run.gravity else ', no self-weight') + ' — '
                     f'step {int(fv["step"][f])}' + ('' if fv['converged'][f] > 0 else f' (staggered iteration {int(fv["staggered_iteration"][f])})') + f', wind q = {load_kpa[f]:.1f} kPa, max d = {fv["d_max"][f]:.2f}, cracked masonry {fv["cracked_volume"][f]:.2f} m³, '
                     f'displacement × {magnify:.0f}', fontsize=11)
        fig.savefig(os.path.join(FRAMES, f'f{k:04d}.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    _, _, fv = load()
    nb = len(selected_frames(fv))
    shutil.rmtree(FRAMES, ignore_errors=True)
    os.makedirs(FRAMES)
    ks = list(range(0, nb, args.every))
    if ks[-1] != nb - 1:
        ks.append(nb - 1)
    with Pool(min(args.workers, len(ks))) as pool:
        pool.map(render, [ks[w::args.workers] for w in range(min(args.workers, len(ks)))])
    # frames are numbered by load step; renumber densely for ffmpeg
    for i, k in enumerate(ks):
        os.rename(os.path.join(FRAMES, f'f{k:04d}.png'), os.path.join(FRAMES, f'g{i:04d}.png'))
    import imageio_ffmpeg
    out = os.path.join(HERE, 'exp_data', f'house_{damage_run.case}.mp4')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-framerate', '6', '-i',
                    os.path.join(FRAMES, 'g%04d.png'), '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p',
                    out], check=True)
    shutil.copy(os.path.join(FRAMES, f'g{len(ks) - 1:04d}.png'), out.replace('.mp4', '_last.png'))
    print(f'wrote {out} ({len(ks)} frames)')
